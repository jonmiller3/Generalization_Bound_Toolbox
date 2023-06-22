import gbtoolbox.bounds as bounds
import gbtoolbox.misc as mt
import numpy as np
import unittest
import time 


class TestSpecNormMethods(unittest.TestCase):
    def test_est_spec_norm_equi_1d(self):
        v = [3.3]
        sng = bounds.spec_norm_gaussian(v)
        N = 1000
        span = 60
        x = np.linspace(-span/2.0,span/2.0,N)[:,None]
        B = N/span

        y = np.exp(-0.5/v[0]*x**2)
        sne = bounds.est_spec_norm_equi(x,y,N,np.array([B]),np.array([[-span/2.0,span/2.0]]))
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,1e-1,msg=msg)

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
        sne = bounds.est_spec_norm_equi(x,y,N,np.array([B]*2),spans)
        rel_er = np.abs(sne-sng)/sng
        msg = f'analytic = {sng}, dft-based = {sne}'
        self.assertLess(rel_er,1e-1,msg=msg)
    
if __name__ == '__main__':
    unittest.main()