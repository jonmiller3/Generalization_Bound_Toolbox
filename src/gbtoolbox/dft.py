from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
from numba import cuda
import math as math


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
