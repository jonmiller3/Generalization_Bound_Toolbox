from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
from numba import cuda
import math as math
try:
    import cupy as cp
except:
    print(" cupy not imported ")

misc_wrapper = CDLL(r'libft.so')

_nu_dft = misc_wrapper.nu_dft
_nu_dft.argtypes = [ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_int,
    c_int,
    ndpointer(np.float64, flags="C"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_int,
    ndpointer(np.complex128, flags="C_CONTIGUOUS"),
]

_nu_dft_e = misc_wrapper.nu_dft_e
_nu_dft_e.argtypes = [ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_int,
    c_int,
    ndpointer(np.float64, flags="C"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    c_int,
    ndpointer(np.complex128, flags="C_CONTIGUOUS"),
]

def threshold_mask(yf: np.array, ns: int, th: float) -> np.array:
    ''' Applies the threshold for approximate FT based on samplesize and
        threshold multiplier.
        
    Args:
        yf: Fourier transform coefficients
        ns: Number of samples
        th: threshold
        
    '''
    yf_max = np.max(np.abs(yf))
    yf_threshold = th*yf_max/np.sqrt(ns)
    mask = np.abs(yf) > yf_threshold
    return mask

def threshold_cmask(yf: np.array, th: float) -> np.array:
    ''' Applies the threshold for approximate FT based on threshold.
        
    Args:
        yf: Fourier transform coefficients
        th: threshold
        
    '''

    mask = np.abs(yf) > th
    return mask

def nu_dft_fast(x: np.array, y: np.array, f: np.array) -> np.array:
    '''Same as nu_dft, but implemented in c. It's 2-3 times faster but fundamentally
    an order n*m operation. You must build the c version using the included Makefile.
    '''
    if (len(y.shape)==1): # common mistake
        y = y.reshape(y.shape[0],1)

    N = x.shape[0]
    D = x.shape[1]
    M = f.shape[0]

    yf = np.zeros((M,1),dtype=np.complex128)    
    _nu_dft(x,c_int(N),c_int(D),y,f,c_int(M),yf)
    return yf

def nu_dft_faster(x: np.array, y: np.array, f: np.array) -> np.array:
    '''Same as nu_dft_fast, but with a tradeoff of more speed for less accuracy. Use 
    with caution. About 2x as fast as nu_dft_fast. You must build the c version
    using the included Makefile.
    '''
    if (len(y.shape)==1): # common mistake
        y = y.reshape(y.shape[0],1)

    N = x.shape[0]
    D = x.shape[1]
    M = f.shape[0]

    yf = np.zeros((M,1),dtype=np.complex128)    
    _nu_dft_e(x,c_int(N),c_int(D),y,f,c_int(M),yf)
    return yf

def nu_dft(x: np.array, y: np.array, f: np.array) -> np.array:
    '''Directly computes in the DFT for non-uniformly sampled inputs
       and non-uniformly sampled frequency vectors

       The computation complexity is of order NxdxM, and is intended for sparse 
       collections of input and output samples in large d cases. 

       Args: 
            x: Nxd array of arbitrary input vectors, each row being a vector of 
               dimension d
            y: Nx1 array of function values at x
            f: A Mxd array of arbitrary frequency vectors
       Returns:
            Mx1 array representing the DFT of the function

    '''
    if (len(y.shape)==1): # common mistake
        y = y.reshape(y.shape[0],1)

    w = -2.0j*np.pi*f
       
    N = x.shape[0]
    M = f.shape[0]
    
    dt = np.complex128 if y.dtype==np.complex128 or y.dtype==np.float64 else np.complex64
    yf = np.zeros((M,1),dtype=dt)
    
    for i in range(M):
        tmp = np.matmul(x,np.transpose(w[i,:]))[:,None]
        yf[i] = np.sum(y*np.exp(tmp))

    # the following is correct, but the array arg gets too big
    # arg = np.matmul(x,np.transpose(w))
    # yf = np.sum(y*np.exp(arg),axis=0)[:,None]
    
    return yf

def dft_on_vector(x,y,u,w):
    '''
    DFT of d-dimensional data at scalar multiples of a unit response frequency vector
    \sum_{i=0}^{n-1}y_ie^{-w[j]u^Tx_i}

    Args:
        x - function input vectors
            Nxd nparray where d = dimension and N = # of samples            
        y - function values at the x samples, Nx1 np array
        u - unit response frequency vector, dx1 nparray
        w - a set of scalars to multiply u by, Mx1 nparray

    Returns:
         Mx1 DFT evaluated at w points along the u veector
    '''
    uTx = np.matmul(x,u).T    
    return np.sum(y.T*np.exp(-1j*w*uTx),axis=1)[:,None]

def nu_sigma_dft(x: np.array, y: np.array, f: np.array):
    '''Directly computes in the DFT for non-uniformly sampled inputs
       and non-uniformly sampled frequency vectors

       The computation complexity is of order NxdxM, and is intended for sparse
       collections of input and output samples in large d cases.
       Args:
            x: Nxd array of arbitrary input vectors, each row being a vector of
               dimension d
            y: Nx1 array of function values at x    
            f: A Mxd array of arbitrary frequency vectors
       Returns:
            Mx1 array representing the sigma for the DFT of the function based on MC theory

    '''
    w = -2.0j*np.pi*f

    N = x.shape[0]
    M = f.shape[0]

    yf = np.zeros((M,1),dtype=np.float64)

    def calc_sigma(xx):
        exx = np.sum(xx)/N
        exx2 = np.sum(np.square(xx))/N
        if exx*exx>exx2:
            return np.sqrt((exx*exx-exx2)/(N-1))
        else:
            return 0


    for i in range(M):
        tmp = np.matmul(x,np.transpose(w[i,:]))[:,None]
        exp_tmp = np.exp(tmp)
        re_tmp = np.real(exp_tmp)
        im_tmp = np.imag(exp_tmp)
        yf[i] = calc_sigma(y*re_tmp)+calc_sigma(y*im_tmp)

    return yf

@cuda.jit
def nu_dft_core(x, y, w, yfr,yfi):
    '''core routine for computing DFT on arbitrary input data. Not generally a user call.
       Args:
            x: Nxd array of arbitrary input vectors, each row being a vector of 
               dimension d
            y: Nx1 array of function values at x
            f: A Mxd array of arbitrary frequency vectors            
            yfr: Real array of output (Numba can't use CUDA to return values)
            yfi: Imaginary array of output    
    '''

    M = w.shape[1]
    N = x.shape[0]
    d = x.shape[1]
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for i in range(i_start,M,threads_per_grid):
        s = 0.0
        c = 0.0
        for n in range(N):
            t = 0.0
            for k in range(d):
                t += x[n,k]*w[k,i]
            c += y[n]*math.cos(t)
            s += y[n]*math.sin(t)
        yfr[i] = c
        yfi[i] = s

def nu_dft_cuda(x: np.array, y: np.array, f: np.array,b = 64, th = 64) -> np.array:
    '''Same as nu_dft, but implemented to use a CUDA GPU. This variant can be 100x faster than
       the nu_dft_fast variant. 
       Args:
            x: Nxd array of arbitrary input vectors, each row being a vector of 
               dimension d
            y: Nx1 array of function values at x
            f: A Mxd array of arbitrary frequency vectors            
            b - number of CUDA blocks
            th - number of CUDA threads per block

    '''
    if (len(y.shape)==1): # common mistake
        y = y.reshape(y.shape[0],1)

    w = -2.0*np.pi*f # no j since using separate real and imaginary parts
    M = f.shape[0]
    
    yt = np.copy(y.flatten())
    wt = np.copy(w.T)

    x_c = cuda.to_device(x)
    y_c = cuda.to_device(yt)
    w_c = cuda.to_device(wt)
    
    yf_cr = cuda.device_array((M,1))
    yf_ci = cuda.device_array((M,1))
    
    nu_dft_core[b,th](x_c, y_c, w_c, yf_cr,yf_ci)
    yfr = yf_cr.copy_to_host()
    yfi = yf_ci.copy_to_host()
    yf = yfr+1j*yfi
    
    return yf


class nu_dft_cupy:
    '''Same fundamental idea of nu_dft, but providing better control and optimzied use of cublas (for float32 and float16). This can be 20x faster than nu_dft_cuda. Returns None if Cupy isn't installed.
    ''' 
    def __init__(self,x,y,gpu_indices=(0,1),nutype='float16',MAXT=256):
        '''Initializes the nu_dft_cupy.
        Args:
            x: The features of the target function. Shape NxD.
            y: The values of the function at each X, shape Nx1.
            gpu_indices: Multi-GPU use is availalbe but not optimized.
            nutype: What precision to use during the matrix multiplication. Recommended 'float16'.
            MAXT: Device and problem specific value to determine best way to setup the cublas kernels. Recommended is 256.
        

        '''     
        try:
            print(" initializing with ",cp.__version__)
        except:
            print(" Cupy not installed ")
            self.N = None
            return

        self.MAXT = MAXT
        self.N = x.shape[0]
        self.d = x.shape[1]
        if nutype=='int8':
            print(" it seems likely that int8 is not optimized in cupy ")
        self.gpu_indices = gpu_indices
        self.n_gpu = len(self.gpu_indices)
        self.streams = [None] * self.n_gpu
        self.x = [None] * self.n_gpu
        self.y = [None] * self.n_gpu
        self.nutype = nutype
        for gpu_id in self.gpu_indices:
            with cp.cuda.Device(gpu_id):
                self.y[gpu_id]=cp.asarray(y.reshape(-1,1).T,dtype='float32')
                self.x[gpu_id]=cp.asarray(x,dtype=self.nutype)
                self.streams[gpu_id] = cp.cuda.stream.Stream()
    
    def process_grid(self,grid_info,threshold,norm,gpu_id,indexing='xy'):
        '''Calculates the approximate Fourier transform on a grid..
        Args:
            grid_info: Information about the grid to set up. Shape dx3. Values are Maximum, minimum, delta.
            threshold: It is recommended to run with a threshold, as most values are consistent with 0.
            norm: Constant used when calculating the approximate Fourier transform.
            gpu_id: What GPU to run on.
            indexing: Recommended 'xy'

        Returns:
            yf: The values of the approximate Fourier transform at the returned frequencies. Shape Mx1. However, if a threshold is set the shape will be reduced to a subset due to values that are consistent with 0 being removed.
            f: The frequencies where the approximate Fourier transform is calculated and where the approxiamte Fourier transform is not consistent with 0 (as determined by the threshold). Shape is Mxd.
            yfu: Unique values for the 1-norm of the frequency, possibly useful in further analysis.
            yfuc: Counts at each unique value for the 1-norm of the frequency, possibly useful in further analysis.
        '''     
        if self.N==None:
            print(" Cupy not installed ")
            return None
        

        with cp.cuda.Device(gpu_id):

            ff = [cp.arange(grid_info[i,0], grid_info[i,1], (grid_info[i,1]-grid_info[i,0])/grid_info[i,2], dtype=self.nutype) for i in np.arange(grid_info.shape[0])]
            ff = cp.meshgrid(*ff, indexing=indexing)
            dd = len(ff)
            NN = ff[0].size
            fff = [cp.reshape(xt,(NN,1)) for xt in ff]
            f = cp.hstack(cp.array(fff))

        cyc, syc = self.process(f,gpu_id)

        with cp.cuda.Device(gpu_id):
            yf = cyc + 1j*syc

            yf = yf*norm

            mff, eff = cp.frexp(np.sum(np.abs(f),axis=1).flatten())
            fsum = cp.ldexp(cp.around(mff,1),eff)
            yfu, yfuc = cp.unique(fsum, return_counts=True)
            tyf = cp.sqrt(yf.real*yf.real+yf.imag*yf.imag)>threshold

        return yf[tyf],f[tyf.flatten(),:],yfu,yfuc
    
    def process(self,f,gpu_id):
        '''Calculates the approximate Fourier transform.
        Args:
            f: Frequencies to calculate the approximate Fourier transform at. Shape Mxd.
            gpu_id: What GPU to run on.

        Returns:
            syc: Imaginary values of the approximate Fourier transform. Shape Mx1.
            cyc: Real values of the approximate Fourier transform. Shape Mx1.
        '''
        
        if self.N==None:
            print(" Cupy not installed ")
            return None


        M = f.shape[0]
        with cp.cuda.Device(gpu_id):
            if self.nutype=='float16':
                wc=cp.asarray(-2*np.pi*f.T,dtype='float16')
            elif self.nutype=='int8':
                wc=cp.asarray(f.T,dtype='int8')
            else:
                wc=cp.asarray(-2*np.pi*f.T,dtype='float32')
                print(" default is float32 ")
            self.streams[gpu_id].use()
            cyc=cp.zeros((1,M),dtype='float32')
            syc=cp.zeros((1,M),dtype='float32')
            if M>self.MAXT:

                for i in cp.arange(int(M/self.MAXT)):

                    if not self.nutype=='int8':
                        wxc=cp.zeros((self.N,self.MAXT),dtype='float32')
                        wxc=cp.matmul(self.x[gpu_id],wc[:,i*self.MAXT:(i+1)*self.MAXT])
                    else:
                        wxc=cp.zeros((self.N,self.MAXT),dtype='int16')
                        wxc=cp.matmul(self.x[gpu_id],wc[:,i*self.MAXT:(i+1)*self.MAXT])
                    self.streams[gpu_id].synchronize()
                    if not self.nutype=='int8':
                        cc=cp.cos(wxc)
                        sc=cp.sin(wxc)
                    else:
                        cc=cp.cos(-1*np.pi*wxc/128)
                        sc=cp.sin(-1*np.pi*wxc/128)
                    self.streams[gpu_id].synchronize()
                    
                    cyc[:,i*self.MAXT:(i+1)*self.MAXT]=cp.matmul(self.y[gpu_id],cc)
                    syc[:,i*self.MAXT:(i+1)*self.MAXT]=cp.matmul(self.y[gpu_id],sc)

                if int(M/self.MAXT)*self.MAXT<M:

                    if not self.nutype=='int8':

                        wxc=cp.matmul(self.x[gpu_id],wc[:,int(M/self.MAXT)*self.MAXT:])
                    else:

                        wxc=cp.matmul(self.x[gpu_id],wc[:,int(M/self.MAXT)*self.MAXT:])

                    self.streams[gpu_id].synchronize()
                    if not self.nutype=='int8':
                        cc=cp.cos(wxc)
                        sc=cp.sin(wxc)
                    else:
                        cc=cp.cos(-1*np.pi*wxc/128)
                        sc=cp.sin(-1*np.pi*wxc/128)
                    self.streams[gpu_id].synchronize()
                    cyc[:,int(M/self.MAXT)*self.MAXT:]=cp.matmul(self.y[gpu_id],cc)
                    syc[:,int(M/self.MAXT)*self.MAXT:]=cp.matmul(self.y[gpu_id],sc)
            else:

                if not self.nutype=='int8':

                    wxc=cp.matmul(self.x[gpu_id],wc)
                else:

                    wxc=cp.matmul(self.x[gpu_id],wc)
                self.streams[gpu_id].synchronize()
                if not self.nutype=='int8':
                    cc=cp.cos(wxc)
                    sc=cp.sin(wxc)
                else:
                    cc=cp.cos(-1*np.pi*wxc/128)
                    sc=cp.sin(-1*np.pi*wxc/128)
                self.streams[gpu_id].synchronize()
                cyc=cp.matmul(self.y[gpu_id],cc)
                syc=cp.matmul(self.y[gpu_id],sc)
            self.streams[gpu_id].synchronize()
            return cyc, syc
                
