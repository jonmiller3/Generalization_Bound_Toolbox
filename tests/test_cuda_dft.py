import gbtoolbox.dft as dft
import gbtoolbox.misc as mt
import numpy as np
import time
import torch

# This is a timing test
def time_it(f,xg,yg,fg,blocks=None,threads=None,approx=False):
    t1 = time.time()
    if blocks == None:
        yf = f(xg,yg,fg)
        dt = time.time()-t1
        print(f'{f.__name__} time : {dt}')
    else:
        # JAM, we should actually check if it is cuda directly
        yf = f(xg,yg,fg,blocks,threads)
        if approx:
            yf = yf[dft.threshold_mask(yf,xg.shape[0],1)]
        dt = time.time()-t1
        print(f'{f.__name__}[{blocks}][{threads}](approx={approx}) time : {dt}')

    return dt
def cor(z1,z2):
    cor1 = np.sum(z1*np.conj(z2))/np.sqrt(np.sum(z1*np.conj(z1))*np.sum(z2*np.conj(z2)))
    return cor1


def gen_1(N,d):
    '''
    Generate random function data for an equispaced d-dimensional input
    '''
    y =  np.random.standard_normal((N,)*d)
    x = np.arange(N,dtype=np.float64)
    xx = [x for i in range(d)]
    xg = mt.grid_to_stack(np.meshgrid(*xx))
    f = np.linspace(0,1,N,endpoint=False)
    ff = [f for i in range(d)]
    fg = mt.grid_to_stack(np.meshgrid(*ff))
    yg = y.flatten()[:,None]
    return xg,yg,fg

def gen_2(N,d,M):
    '''
    Generate random function data for random d-dimensional input
    '''
    yg =  np.random.standard_normal((N,1))
    xg = np.random.random((N,d))
    f = np.linspace(0,1,M,endpoint=False)
    ff = [f for i in range(d)]
    fg = mt.grid_to_stack(np.meshgrid(*ff))
    return xg,yg,fg


# equispaced samples example
N =64 # samples per dimension
d = 2 # number of dimensions
xg,yg,fg = gen_1(N,d)

print('Equispaced Samples')

dt1 = time_it(dft.nu_dft_fast,xg,yg,fg)
dt2 = time_it(dft.nu_dft_faster,xg,yg,fg)
if torch.cuda.is_available():
    dt3 = time_it(dft.nu_dft_cuda,xg,yg,fg,(N**d)//1024,256)
    dt4 = time_it(dft.nu_dft_cuda,xg,yg,fg,N,N)
    dt5 = time_it(dft.nu_dft_cuda,xg,yg,fg,N,N,True)

# arbitrary samples example
N = 1000
d = 6
M = 16
xg,yg,fg = gen_2(N,d,M)

print('Arbitrary Samples')

dt1 = time_it(dft.nu_dft_fast,xg,yg,fg)
dt2 = time_it(dft.nu_dft_faster,xg,yg,fg)
if torch.cuda.is_available():
    dt3 = time_it(dft.nu_dft_cuda,xg,yg,fg,(M**d)//1024,256)
    dt4 = time_it(dft.nu_dft_cuda,xg,yg,fg,M**(d-1),M)
    dt5 = time_it(dft.nu_dft_cuda,xg,yg,fg,M**(d-1),M,True)

                          
