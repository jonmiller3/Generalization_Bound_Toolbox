import numpy as np
import gbtoolbox.dft as dft
import gbtoolbox.misc as mt
from ctypes import *
from numpy.ctypeslib import ndpointer
import scipy.optimize as opt
from geomloss import SamplesLoss


ztw_c = CDLL(r'libztw.so')


    

def est_spec_norm_from_data(x: np.array, y: np.array, B: np.array, f: np.array, S: np.array, nu_type='nu_dft_fast', threshold=0.0) -> float:
    '''Estimate the spectral norm from a set of inputs/ouputs
       given a set of frequencies at which to evaluate the Fourier transofrm

       Args:
            x: The inputs to the function, a Nxd array, where each row corresponds
               to an input vector            
            y: A size Nx1 array of outputs for the form y = f(x)
            B: The bandwidths in each dimension
            f: Kxd array of frequency vectors at which to evaluate the Fourier transform
            threshold: Threshold for approximate FT. This is currently based on threshold*max(abs(yf))/sqrt(N) where yf is Fourier Transform
    '''
    algs = {'nu_dft':dft.nu_dft,'nu_dft_fast':dft.nu_dft_fast,'nu_dft_faster':dft.nu_dft_faster,'nu_dft_cuda':dft.nu_dft_cuda}
    alg = algs[nu_type]
    d = x.shape[1]        

    dS = S[:,1]-S[:,0]
    V = np.prod(dS)

    yf = (V/x.shape[0])*alg(x,y,f)/(np.sqrt(2*np.pi)**d)
    mask =dft.threshold_mask(yf,x.shape[0],threshold)
    
    two_pi =2.0*np.pi
    return est_spec_norm(f*two_pi,yf,B*two_pi,mask)

def est_spec_norm_random(x: np.array, y: np.array, K: int, B: np.array, S: np.array, nu_type='nu_dft_fast', threshold=0.0) -> float:
    '''Estimate the spectral norm from a set of inputs/ouputs
       that represent a function using random frequency vectors

       Args:
            x: The inputs to the function, a Nxd array, where each row corresponds
               to an input vector
            y: A size Nx1 array of outputs for the form y = f(x)
            K: The number of frequency vectors to randomly select
            B: The bandwidths in each dimension
            S: The domain of the inputs, a dx2 array
            threshold: Threshold for approximate FT. This is currently based on threshold*max(abs(yf))/sqrt(N) where yf is Fourier Transform
    '''
    d = x.shape[1]    
    f = np.random.random_sample((K,d))-0.5
    return est_spec_norm_from_data(x, y, B, f, S,nu_type,threshold)

def est_spec_norm_f(domain: np.array, f, N:int) -> float:
    '''Estimate the spectral norm of a function f given its domain in each dimension
       and the number of points in each dimension to sample f from

       Note this function should only be used for small dimension size
       
       Note that this requires an analytic function f

        Args:
            domain: dx2 array that describes the domain of the function (a hyper rectangle)
            f: a function of d variables with scalar outputs
            N: The number of points along each dimension to sample. So N**d total
            total points will be generated.

        Returns:
            an estimate of the spectral norm of the input function
            

    '''
    d = domain.shape[0]
    xx = [np.linspace(dt[0],dt[1],N,endpoint=False) for dt in domain]
    x = np.meshgrid(*xx)
    x = [xt.flatten() for xt in x]
    y = np.array([f(*xt) for xt in zip(*x)]).reshape(*((N,)*d))
    V = np.prod(domain[:,1]-domain[:,0])
    yf = V/(N**d)*np.fft.fftshift(np.fft.fftn(y))
    B = np.array([2.0*np.pi/(xt[1]-xt[0]) for xt in xx]) # radial bandwidths assuming Nyquist

    ww = [np.linspace(-0.5,0.5,N,endpoint=False)*Bt for Bt in B]
    w = np.meshgrid(*ww)
    wn2 = np.zeros((N,)*d)
    for wi in w:
        wn2 += np.abs(wi)
    wn2 = wn2*wn2
    sn = np.prod(B)/(N**d)*np.sum(wn2*np.abs(yf))
    return sn

def est_spec_norm_equi(x: np.array, y: np.array, M: int, B: np.array, S: np.array, nu_type='nu_dft_fast', threshold=0.0) -> float:
    '''Estimate the spectral norm from a set of inputs/ouputs
       that represent a function using equispaced frequencies

       Args:
            x: The inputs to the function, a Nxd array, where each row corresponds
               to an input vector
            y: A size Nx1 array of outputs for the form y = f(x)
            M: The number of equispaced frequencies in each dimension
            B: The bandwidths in each dimension
            S: The domain of the inputs, a dx2 array
            threshold: Threshold for approximate FT. This is currently based on threshold*max(abs(yf))/sqrt(N) where yf is Fourier Transform

    '''
    
    d = x.shape[1]
    Bspans = np.array([[-Bt/2.0,Bt/2.0] for Bt in B])
    f, _ = mt.gen_stacked_equispaced_nd_grid(M,Bspans)
    
    return est_spec_norm_from_data(x, y, B, f, S,nu_type,threshold)

def est_spec_norm(w: np.array, yf: np.array, B=None, mask=None) -> float:
    '''Estimate the spectral norm given a finite set of frequeny vectors 
       and estimates of the Fourier transform at those vectors

       The spectral norm of a continuous function is the definite integral of the product 
       of the absolute value of the Fourier transform and the one-norm squared of the
       frequency vector.
       
       Args:
            w: M x d array of radial frequency vectors, were d is the dimension and M is the number 
              of vectors. These values are in the continuous domain, (i.e. not limited to [-pi,pi])            
            yf: length M array of Fourier transform values
            B: radial bandwidths in each dimension
            mask: M array of boolians for whether the Fourier transform valeus are included in the approximate Fourier transform or not.
       Returns:
           An estimate of the spectral norm
       '''
       
       
    d=w.shape[1]
    
    assert w.shape[0]==yf.shape[0], " shapes mismatch, w should be Mxd and yf should be M"
    
    w2 = (np.sum(np.abs(w),axis=1)**2)[:,None] # 1 norm squared
    if mask is not None:
        if not mask.all():
            w2yf = w2*np.abs(yf)*mask.astype(int)
            frac_in_domain = sum(mask.astype(int))/len(mask)
        else:
            w2yf = w2*np.abs(yf)
            frac_in_domain=1.
    else:
        w2yf = w2*np.abs(yf)
        frac_in_domain=1.

    #B = None

    # compute the bandwidth volume
    if B is None:
        mw = np.min(w,axis=0)
        Mw = np.max(w,axis=0)
        V = np.prod(Mw-mw)
    else:
        V = np.prod(B)
    
    
    # JAM, should this be w2yf.shape[0]
    fac = V/yf.shape[0] # volume over total number of points
    return np.sum(w2yf)*fac*np.sqrt(2*np.pi)**d

def spec_norm_gaussian(var) -> float:
    '''Computes the analytical spectral norm of an n-d unscaled Gaussian
    
       Unscaled simply means that the term in front of the exponential is omitted from the Gaussian.
       i.e. a function of the form e^{-0.5*sum(x**2/s**2)}, where s is the standard deviation

       Args:
            var: a list or array of variances, one for each dimension
       Returns:
            An analytical computation of the spectral norm    
    '''    
    var = np.array(var)
    d = len(var)    
    ivar = 1/var
    s1 = np.sum(ivar)
    
    istd = np.sqrt(ivar)
    s2 = 0.0
    for i in range(d):
        s2 += np.sum(istd[i]*istd[i+1:])
    sn = (2.0*np.pi)**d*(s1+4.0/np.pi*s2)
    return sn

def spec_norm_sin(var) -> float:
    '''Computes the analytical spectral norm of an n-d product of sins, \prod sin(n_i x_i)


       Args:
            var: a list or array of n_i, one for each dimension
       Returns:
            An analytical computation of the spectral norm
    '''
    var = np.array(var)
    d = len(var)
    w2 = 0

    for i in range(d):
        w2 += np.abs(var[i])
    w2 = w2*w2
    
    sn = (2.0*np.pi)**d*w2
    return sn

def path_norm_2layer(weights1,biases1,weights2,bias2=0):
    '''
    Calculate the path norm of a NN with one hidden layer and a single output node

    Args:
        weights1: NxM array of weights for the hidden layer
                  N is the number of inputs
                  M the number of nodes in the hidden layer
        biases1: length M array of biases
        weights2: Mx1 array of weights for the output layer
        bias2: The bias of the output layer

    '''
    wnorm = np.sum(np.abs(weights1),axis=0) # 1-norm of weights
    p_norm = np.sum(np.abs(weights2)*(wnorm+np.abs(biases1)))+np.abs(bias2)
    return p_norm


class TwoLayerNetwork:
    def __init__(self, inner_weights, inner_biases, outer_weights):   
        '''
        Initialize two layer network (one hidden layer). A two layer network is of
        the form \sum_i c_i\sigma(w_i^Tx+b_i)

        Args:

            inner_weights - d x M  array representing M d-dimensional weigths w_i
            inner_biases - 1 x M array representing scalar biases b_i
            outer_weights - 1 x M array representing scalar weights c_i
        '''             
        self.w = inner_weights
        self.b = inner_biases
        self.c = outer_weights
    
    def relu(self, x):
        return np.maximum(0,x)
    
    def evaluate(self,x):           
        '''
        Evaluate the network at a set of inputs
        Args:
            x: Nxd array of N d-dimensional samples
        Return:
            A length N array of outputs
        ''' 
        y = np.sum(self.c*self.relu(x@self.w+self.b),axis=1)
        return y
    
    def path_norm(self):
        '''
        Calculate the path norm of the network
        '''

        wnorms = np.sum(np.abs(self.w),axis=0) # 1-norms of weight vectors
        p_norm = np.sum(np.abs(self.c)*(wnorms+np.abs(self.b)))
        return p_norm
    
        


def apriori_bound(spectral_norm: float, width: float, sample_size: float, dimension: float,
                confidence: float):
    '''
    Calculate the bound for the x-values given using derivation based on E's paper
    Estimate the bound on the expected square error of the output of the neural
    network found by optimizing the path-norm regularized loss of E et al. That is, the 
    error is a combination of approximaton and estimation error. 

    This bound assumes that the spectral norm is greater than 1.

    Args:
        spectral_norm: spectral norm of the function
        width: width of shallow neural network
        sample_size: number of samples the network has been trained on
        dimension: dimension of the data
        confidence: confidence interval, 1 - delta (must be between 0 and 1, exclusive)
    
    Returns:
        A number bounding the expected value of the squared error
    '''
    c = 3.289868133696453 #2*np.pi**2/6
    cidelta = c/(1.0 - confidence)
    iss = 2.0/sample_size
    greek_lambda = 4.0*np.sqrt(np.log(2.0*dimension)*iss)
    igreek_lambda = 1.0/greek_lambda
    sn2 = spectral_norm*spectral_norm
    iwidth = 1.0/width

    b1 =  0.5*np.sqrt(np.log(4.0*sn2*cidelta)*iss)
    t = 3.0*sn2*iwidth
    theta_bound = t*igreek_lambda + 4.0*spectral_norm + igreek_lambda*b1

    return t + 4.0*greek_lambda*spectral_norm + b1 + 0.5*np.sqrt(np.log(
           theta_bound*theta_bound*cidelta)*iss)
    # TLR,
    # the following is a direct implementation and is substantially slower
    # c = np.pi**2/6
    # delta = 1 - confidence
    # greek_lambda = 4*np.sqrt(2*np.log(2*dimension)/sample_size)

    # TLR, the bounds below seems to be approximations to eqs. 152 and 153 from
    # interim report. The approximation is that gamma is >> 1. 
    # theta_bound = (3*spectral_norm**2/(width*greek_lambda) + 4*spectral_norm + 1/(2*greek_lambda)*
    #                np.sqrt((2*np.log(8*c*spectral_norm**2/delta))/sample_size))

    # TLR, There seems to be another assumption that theta_bound >> 1. Both assumptions
    # seem reasonable.
    # return 3*spectral_norm**2/width + 4*greek_lambda*spectral_norm + 1/2*np.sqrt(2*np.log(
    #        8*c*spectral_norm**2/delta)/sample_size) + 1/2*np.sqrt(2*np.log(
    #        2*c*theta_bound**2/delta)/sample_size)


def nn_wnorm(weights):
    '''
    
    Args:
        inner_weights - d x M array of inner weights
    Returns:
        A numpy array of size d X M of inner weights divided by the 1-norm of the inner weights.
        The 1-norm of the inner weights.
                        
    '''
    # JAM, Is this used somewhere?
    a = weights
    na = np.sum(np.abs(a),axis=0) # 1-norm
    th = np.multiply(a,1./na)
    
    return th,na

_gen_ztw = ztw_c.gen_ztw
_gen_ztw.argtypes = [
    c_int,# m
    c_int,# d
    c_int,# k
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# zv
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# wv
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# tv
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# sv
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# w
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# wftm
    ndpointer(np.float64, flags="C_CONTIGUOUS"),# wfta
    ndpointer(np.float64, flags="C_CONTIGUOUS")# magw
    
]

class E_pdf:
    def __init__(self, FT, w):   
        '''
        Initialize class with Fourier tranform

        Args:
            FT - Fourier tranform values, length nx1 array, where n is the number of
                 frequency vectors
            w - Radial frequency vectors at which the FT is computed
                dxn array where d is the dimension of the vectors
                and n is the number of frequency vectors
                Note that if you use f from the toolbox, you need to provide f*2*np.pi
            

        '''             
        magw = np.sum(np.abs(w),axis=0).flatten()        

        aFT = np.abs(FT)       

        # TLR,
        # the following 4 are only used for display, can be removed at
        # some point if needed
        self.weighted_FTmag = np.multiply(aFT,np.multiply(magw,magw).reshape(-1,1))
        self.FTangle = np.angle(FT)
        self.w = w
        self.magw = magw
        # TLR,
        # no need to deal with 0 values, so get rid of em
        inds = np.where(aFT>0)[0]
        self.weighted_FTmag_f = self.weighted_FTmag[inds]
        self.FTangle_f = self.FTangle[inds]
        self.w_f = self.w[:,inds]
        self.magw_f = self.magw[inds]

    @staticmethod
    def mag_thresh(y, tp = 0.95):           
        '''
        Calculate a threshold to retain the top fraction of magnitude in the input
        Args:
            y - absolute value of Fourier transform
            tp - top fraction of magnitude to keep (e.g. 0.95 means that we find
            the set of the highest values such that their sum is 95% of the sum
            over all values)

        '''
        # JAM, this should in general be handled at the Fourier transform stage
        # since that is where you have the required information
        sy = np.sort(y)[::-1] # descending order
        csy = np.cumsum(sy)
        tot = csy[-1]
        ind = np.where(csy>=tp*tot)[0][0]
        thresh = sy[ind]
        return thresh
    
    def gen_ztw(self,m):
        '''
        Randomly generate m instances of z, t, and w from E's probability
        distribution

        Args:
            m - the number of random draws to make
        Return:
            z - length m array of values from the set {-1,1}
            t - length m array of values from [0,1]
            w - size d x m array of frequency vectors

        Note: uses the rejection sampling method to produce random data with desired
        distibution
        '''
        d = self.w_f.shape[0]
        k = self.w_f.shape[1]
        magw = self.magw_f
        wftm = self.weighted_FTmag_f/np.max(self.weighted_FTmag_f)

        b = self.FTangle_f
        w = self.w_f

        zv = np.zeros(m)
        tv = np.zeros(m)
        wv = np.zeros((d,m)) 
        sv = np.zeros(m)
        cnt = 0
        while cnt < m:
            t = np.random.random()
            z = np.random.randint(0,2)*2.0-1.0
            w_ind = np.random.randint(k)
            tmp = np.cos(magw[w_ind]*t-z*b[w_ind])
            p = wftm[w_ind]*np.abs(tmp)            
            
            chance = np.random.random()
            if chance < p:
                zv[cnt] = z
                tv[cnt] = t
                wv[:,cnt] = w[:,w_ind]
                sv[cnt] = -np.sign(tmp)
                cnt +=1

        return zv,tv,wv,sv

    def gen_ztw_c(self,m):
        '''
        Same thing as gen_ztw, but implemented using c and far faster.
        '''
        d = self.w_f.shape[0]
        k = self.w_f.shape[1]
        magw = self.magw_f
        wftm = self.weighted_FTmag_f/np.max(self.weighted_FTmag_f)

        b = self.FTangle_f
        w = self.w_f

        zv = np.zeros(m)
        tv = np.zeros(m)
        wv = np.zeros((d,m))
        sv = np.zeros(m)
        wt = np.copy(self.w_f.T,order='C').flatten()

        _gen_ztw(m,d,k,zv,wv,tv,sv,wt,wftm,b,magw)

        return zv,tv,wv,sv

    def gen_nn(self,m):        
        '''
        Generate a two-layer neural network via sampling the PDF
        Args:
            m: number of nodes in hidden layer
        Return
            A TwoLayerNetwork object
        '''
        
        # JAM, we need to divide s by spectral norm
        z,t,w,s=self.gen_ztw_c(m) # z,t, and w comprise the parameters vector theta used in math descriptions
        sw = np.sum(np.abs(w),axis=0)[None,:]
        nw = w*(z*(1/sw))        
        nn = TwoLayerNetwork(nw,-t,s)
        return nn
        
def wasserstein_metric(samples_a: np.array, samples_b: np.array) -> float:
    '''Estimates the d-dimensional Wasserstein (2) metric between two distributions given samples
       from each distribution. That is, what is calculated is the Wasserstein metric between the
       empirical distributions formed from the samples.

    Args:
        samples_a, samples_b: N x d arrays representing N points in d dimensions

    Output:
        The square root of the minimum sum of the squared distances between the two
        sets of points
    '''
    # TLR,
    # First calculate all pairwise distances
    n_pts = samples_a.shape[0]
    cost = np.zeros((n_pts, n_pts)) # Where the distances will be stored
    for index_a in range(n_pts):
        for index_b in range(n_pts):
            cost[index_a, index_b] = np.dot(samples_a[index_a] - samples_b[index_b],
                                               samples_a[index_a] - samples_b[index_b])                
    
    rows,cols = opt.linear_sum_assignment(cost)
    total = cost[rows, cols].sum()
    return np.sqrt(total/len(rows))

def sinkhorn_metric(samples_a: np.array, samples_b: np.array, blur: float = 0.05) -> float:
    '''Approximates the d-dimensional Wasserstein metric between two distributions given 
       samples from each distribution. Uses a Kullback-Leibler regularization which makes the 
       assignment problem convex and quickly solvable. The closer the blur value is to 0, the 
       close the result will be to the Wasserstein metric.
    
    Args: 
        samples_a, samples_b: N x d arrays representing N sample points in d dimensions
        blur - a blurring factor, the closer to 0 the close to Wasserstein and slower convergence

    Output:
        The Sinkhorn metric between two sets of samples pulled from distributions.


    '''
    a = torch.tensor(samples_a)
    b = torch.tensor(samples_b)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur)     
    return np.sqrt(2.0*loss(a, b))
