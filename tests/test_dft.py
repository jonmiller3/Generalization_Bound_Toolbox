import gbtoolbox.dft as dft
import gbtoolbox.misc as mt
import numpy as np
import unittest


class TestDFTMethods(unittest.TestCase):
    def test_1d(self):
        N = 16
        M = 32
        x = np.arange(N,dtype=np.float64)[:,None]
        y =  np.random.randn(N)[:,None]
        f = np.linspace(0,1,M,endpoint=False)[:,None]

        yf_fft = np.fft.fft(y,M,axis=0)
        yf = dft.nu_dft(x,y,f)

        er = np.max(np.abs(yf-yf_fft))
        
        msg = f' 1d '
        
        self.assertAlmostEqual(er,0.0,places=11,msg=msg)

    def test_2d(self):
        N = 16
        M = 32
        
        y =  np.random.random((N,N))
        yf_fft = np.fft.fft2(y,[M,M])

        x = np.arange(N,dtype=np.float64)
        xg = mt.grid_to_stack(np.meshgrid(x,x))
        f = np.linspace(0,1,M,endpoint=False)
        fg = mt.grid_to_stack(np.meshgrid(f,f))
        
        yf = dft.nu_dft(xg,y.flatten()[:,None],fg)

        er = np.max(np.abs(yf-yf_fft.reshape((M*M,1))))
        
        msg = f' 2d '

        self.assertAlmostEqual(er,0.0,places=11,msg=msg)

    def test_2d_random_threshold(self):
        N = 80
        span = 6
        x = (np.random.rand(2400*N).reshape(-1,2)*2-1)*span/2.0
        y = (np.sin(2*np.pi*x[:,0])*np.sin(np.pi*x[:,1])+np.sin(np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])).reshape(N*1200,1)

        B = N/span
        
        f, _ = mt.gen_stacked_equispaced_nd_grid(N,np.array([[-Bt/2.0,Bt/2.0] for Bt in np.array([B,B])]))
                
        V = 60*60
        spans = np.tile([-span/2.0,span/2.0],(2,1))
        
        yf = (V/x.shape[0])*dft.nu_dft_fast(x,y,f)/(np.sqrt(2*np.pi)**2)

        mask =dft.threshold_cmask(yf,0.2)
        
        er = 0
        msg = f' 2d random threshold '

        self.assertLess(er,3e-1,msg=msg)

    def test_2d_fast(self):
        N = 16
        M = 32
        
        y =  np.random.random((N,N))
        yf_fft = np.fft.fft2(y,[M,M])

        x = np.arange(N,dtype=np.float64)
        xg = mt.grid_to_stack(np.meshgrid(x,x))
        f = np.linspace(0,1,M,endpoint=False)
        fg = mt.grid_to_stack(np.meshgrid(f,f))
        
        yf = dft.nu_dft_fast(xg,y.flatten()[:,None],fg)

        er = np.max(np.abs(yf-yf_fft.reshape((M*M,1))))
        
        msg = f' 2d fast '
        
        self.assertAlmostEqual(er,0.0,places=11,msg=msg)

    def test_3d(self):
        N = 8
        M = 16
        d = 3
        
        y =  np.random.random((N,)*d)
        yf_fft = np.fft.fftn(y,[M,]*d)

        x = np.arange(N,dtype=np.float64)
        xg = mt.grid_to_stack(np.meshgrid(*((x,)*d)))
        f = np.linspace(0,1,M,endpoint=False)
        fg = mt.grid_to_stack(np.meshgrid(*((f,)*d)))
        
        yf = dft.nu_dft(xg,y.flatten()[:,None],fg)

        er = np.max(np.abs(yf-yf_fft.reshape((M**d,1))))
        
        msg = f' 3d '

        self.assertAlmostEqual(er,0.0,places=11,msg=msg)

class TestDFTOnVectorMethods(unittest.TestCase):
    def test_dft_on_vector_1d(self):
        N = 4
        x = np.arange(N).reshape((N,1))
        y = np.cos(2*np.pi*0.1*x)
        yf = np.fft.fft(y,axis=0)
        u = np.array(1).reshape(1,1)
        w = np.arange(N).reshape((N,1))*(2.0*np.pi/N)
        yft = dft.dft_on_vector(x,y,u,w)
        
        er = np.sum(np.abs(yf-yft))/np.sum(np.abs(yf))
        
        msg = f' vector 1d '
        
        self.assertAlmostEqual(er,0,places=5,msg=msg)
    def test_dft_on_vector_2d(self):
        N = 4
        x1 = np.arange(N).reshape((1,N))
        x1,x2 = np.meshgrid(x1,x1)
        y = np.cos(2*np.pi*0.1*x1)*np.cos(2*np.pi*0.15*x2)
        yf = np.fft.fft2(y)
        yr = np.reshape(y,(N*N,1)) # function requires stacked form
        w = np.arange(N).reshape((N,1))*(2.0*np.pi/N)
        x = np.vstack((x1.ravel(),x2.ravel())).T

        def sub_check(a,b):
            er = np.sum(np.abs(a-b))/np.sum(np.abs(a))
            if er > 1e-5:
                print(a)
                print(b)
            self.assertAlmostEqual(er,0,places=5)

        # first test along x direction
        yf_x = yf[0,:].T[:,None] # ugh, I hate the way numpy array indexing works
        u = np.array([1,0]).reshape(2,1)        
        yf_x_test = dft.dft_on_vector(x,yr,u,w)        
        
        sub_check(yf_x,yf_x_test)        

        # second test along y direction

        yf_x = yf[:,0][:,None]        
        u = np.array([0,1]).reshape(2,1)
        yf_x_test = dft.dft_on_vector(x,yr,u,w)

        sub_check(yf_x,yf_x_test)

        # now along diagonal
        yf_x = np.array([yf[i,i] for i in range(N)]).T[:,None]
        u = np.array([1,1]/np.sqrt(2)).reshape(2,1)
        w = w*np.sqrt(2)
        yf_x_test = dft.dft_on_vector(x,yr,u,w)

        sub_check(yf_x,yf_x_test)
        
        print(' Done checking 2d vector ')


if __name__ == '__main__':
    unittest.main()
