import gbtoolbox.bounds as bounds
import gbtoolbox.misc as mt
import gbtoolbox.dft as dft
import numpy as np
import unittest
import time

import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.interpolate import RBFInterpolator

class TestSpecNormMethods(unittest.TestCase):
    def test_est_spec_norm_equi_1d(self):
        
        v = [3.3]
        sng = bounds.spec_norm_gaussian(v)
        N = 1000
        span = 60
        x = np.linspace(-span/2.0,span/2.0,N)[:,None]
        B = N/span

        y = np.exp(-0.5/v[0]*x**2)
        sne = bounds.est_spec_norm_equi(x,y,int(1*N),np.array([1*B]),np.array([[-span/2.0,span/2.0]]))
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,1e-1,msg=msg)

    def test_est_spec_norm_rand_1d(self):
    
    
        v = [3.3]
        sng = bounds.spec_norm_gaussian(v)
        N = 10000
        span = 60
        x = (np.random.rand(N).reshape(-1,1)*2-1)*span/2.0
        B = 4

        y = np.exp(-0.5/v[0]*x**2)
        sne = bounds.est_spec_norm_equi(x,y,int(10*N),np.array([(B)]),np.array([[-span/2.0,span/2.0]]),'nu_dft_fast',8.0)
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,3e-1,msg=msg)

    def test_est_spec_norm_equi_2d(self):
    
    
        v = [3.3,3.8]
        sng = bounds.spec_norm_gaussian(v)
        N = 80
        span = 60
        x = np.linspace(-span/2.0,span/2.0,N)[:,None]
        X = np.meshgrid(x,x)
        x = mt.grid_to_stack(X)
        y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1])).reshape(N*N,1)
        
        B = N/span

        spans = np.tile([-span/2.0,span/2.0],(2,1))
        sne = bounds.est_spec_norm_equi(x,y,N,np.array([B]*2),spans,'nu_dft_fast')
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,1e-1,msg=msg)

    def test_est_spec_norm_rand_2d(self):
    
    
        v = [3.3,3.8]
        sng = bounds.spec_norm_gaussian(v)
        N = 80
        span = 40
        x = (np.random.rand(1000*N).reshape(-1,2)*2-1)*span/2.0
        y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1])).reshape(N*500,1)
        
        B = N/span

        spans = np.tile([-span/2.0,span/2.0],(2,1))
        sne = bounds.est_spec_norm_equi(x,y,N,np.array([B]*2),spans,'nu_dft_fast',16.0)
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,4e-1,msg=msg)

    def test_est_spec_norm_rand_2d_explicit(self):
    
    
        v = [3.3,3.8]
        sng = bounds.spec_norm_gaussian(v)
        N = 80
        span = 40
        x = (np.random.rand(3000*N).reshape(-1,2)*2-1)*span/2.0
        y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1])).reshape(N*1500,1)
        
        B = N/span

        
        f, _ = mt.gen_stacked_equispaced_nd_grid(N,np.array([[-Bt/2.0,Bt/2.0] for Bt in np.array([B,B])]))
        

        V = span*span
        spans = np.tile([-span/2.0,span/2.0],(2,1))

        
        yf = (V/x.shape[0])*dft.nu_dft_fast(x,y,f)/(np.sqrt(2*np.pi)**2)
        
        B=B*2.0*np.pi
        f=f*2.0*np.pi

        mask =dft.threshold_cmask(yf,0.28)
        
        
        sne = bounds.est_spec_norm(f,yf,None,mask)

        #sne = bounds.est_spec_norm(f,yf,np.array([B,B]),mask)
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,4e-1,msg=msg)

    def test_est_spec_norm_equi_3d(self):
    
    
        v = [3.3,3.8,3.1]
        sng = bounds.spec_norm_gaussian(v)
        N = 100
        span = 20
        x = np.linspace(-span/2.0,span/2.0,N)[:,None]
        X = np.meshgrid(x,x,x)
        x = mt.grid_to_stack(X)
        y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1]+x[:,2]**2/v[2])).reshape(N*N*N,1)
        
        
        B = N/span

        spans = np.tile([-span/2.0,span/2.0],(3,1))
        sne = bounds.est_spec_norm_equi(x,y,N,np.array([B]*3),spans,'nu_dft_cuda')
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,5e-1,msg=msg)
        
    def test_est_spec_norm_rand_3d_explicit(self):
    

    
        v = [3.3,3.8,3.1]
        sng = bounds.spec_norm_gaussian(v)
        N = 80
        span = 50
        x = (np.random.rand(18000*N).reshape(-1,3)*2-1)*span/2.0
        y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1]+x[:,2]**2/v[2])).reshape(N*6000,1)
        
        
        
        f, _ = mt.gen_stacked_equispaced_nd_grid(N,np.array([[-Bt/2.0,Bt/2.0] for Bt in np.array([2,2,2])]))
        
        
        V = span*span*span
        spans = np.tile([-span/2.0,span/2.0],(3,1))

        
        yf = (V/x.shape[0])*dft.nu_dft_cuda(x,y,f,128,128)/(np.sqrt(2*np.pi)**2)
        mask =dft.threshold_cmask(yf,2.0)
        
        f=f*2.0*np.pi
        
        #sne = bounds.est_spec_norm(f,yf,np.array([2,2,2]),mask)
        sne = bounds.est_spec_norm(f,yf,None,mask)
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,6e-1,msg=msg)

    def test_est_spec_norm_rand_rand_6d_explicit(self):
    

        v = [3.3,3.8,4.2,4.6,5.1,6.4]
        sng = bounds.spec_norm_gaussian(v)
        N = 2000000
        span = 40
        x = (np.random.rand(N*6).reshape(-1,6)*2-1)*span/2.0
        y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1]+x[:,2]**2/v[2]+x[:,3]**2/v[3]+x[:,4]**2/v[4]+x[:,5]**2/v[5])).reshape(N,1)
        
        d = x.shape[1]
        f = np.random.random_sample((2000000,d))-0.5
        
        
        V = span*span*span*span*span*span
        spans = np.tile([-span/2.0,span/2.0],(6,1))

        
        yf = (V/x.shape[0])*dft.nu_dft_cuda(x,y,f,256,128)/(np.sqrt(2*np.pi)**5)
        #mask =dft.threshold_mask(yf,x.shape[0],16.0)
        mask =dft.threshold_cmask(yf,85.0)

        
        f=f*2.0*np.pi

        sne = bounds.est_spec_norm(f,yf,None,mask)
        #sne = bounds.est_spec_norm(f,yf,np.array([4,4,4,4,4,4]),mask)
        rel_er = np.abs(sne-sng)/sng
        print(" here is the results {} {} {}".format(sng,sne,rel_er))
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,3e0,msg=msg)

    def test_opt_bound(self):


        # This test only checks if the function runs and produces an answer
        # the right form. It doesn't check for a correct answer. More work is needed
        # to produce an actual numeric test. 
        N = 100
        d = 2        
        m = 60
        trials = 3
        Nd = 8
        B = 2.5        

        U = np.random.random

        # create a random two-layer network
        w = U((d,m))
        b = U((1,m))
        ow = U((1,m))
        nn = bounds.TwoLayerNetwork(w,b,ow)

        # create some random input
        x = U((N,d))

        # use the network as the ideal function
        y = nn.evaluate(x)
        # y = np.exp(-0.5*(x[:,0]**2/v[0]+x[:,1]**2/v[1])).reshape(N*1500,1)


        # TBD: perhaps a smooth function like (1-cos(x1)) can be used along
        # with a NN approximation so that est_bounds can be more appropriately
        # tested. 
        
        tot,ap,opt,optb = bounds.est_bounds(x,y,m,trials,Nd,B,nn)
        
        print(f'a priori error : {ap:5.2f}')
        print(f'total error {tot:5.2f}')
        print(f'Optimization error : {opt:5.2f}')
    
if __name__ == '__main__':
    unittest.main()
