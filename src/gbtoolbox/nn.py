import numba
from numba import jit, types, typed
from numba import int32, float32    # import the types                                                                                                                      
from numba.experimental import jitclass
import numpy as np
import math
try:
    import cupy as cp
except:
    print(" cupy not imported ")

#kv_ty = (types.unicode_type,float32[:])                                                                                                                                    

#spec = [('params', types.containers.UniTuple(float32[:], 4)),('cache', d),('input_size', int32),('neurons_num', int32)]                                                    

#@jitclass(spec)                                                                                                                                                            
class NeuralNet():

    def __init__(self, input_size, neurons_num, bool_barron = False):
        self.cache = {} #{"X":np.zeros(32),"W0":np.zeros((input_size,neurons_num))}                                                                                         
        self.params = {} # (np.zeros((input_size,neurons_num)),np.zeros((1,neurons_num)),np.zeros((1,neurons_num)),np.zeros((1,1)))                                         
        self.input_size = input_size
        self.neurons_num = neurons_num
        self.params = self.init(self.input_size, self.neurons_num)
        self.barron = bool_barron

    def init(self, input_size, neurons_numb):
        scale = 1/max(1., (input_size+input_size)/2.)
	limit = np.sqrt(3.0 * scale)
        oscale = 1/max(1., (neurons_numb+1)/2.)
        olimit = np.sqrt(3.0 * oscale)
        istd = np.sqrt(2.0 / input_size)
        ostd = np.sqrt(2.0 / neurons_numb)
        W0 = np.empty((neurons_numb,input_size),dtype=np.float64)
        W1 = np.empty((1,neurons_numb),dtype=np.float64)
        b0 = np.empty((neurons_numb,1),dtype=np.float64)
	for i in range(neurons_numb):
            b0[i,0]=(np.random.rand()*2-1)*istd
            W1[0,i]=np.random.randn()*ostd
            #W1[0,i]=(np.random.rand()*2-1)*olimit                                                                                                                          
            for k in range(input_size):

            
                b1 = np.zeros((1,1))
                return W0,b0,W1,b1,W1
            
    def barron_init(self,input_size,W0,b0,W1,b1):
        self.params = W0,b0,W1,b1,W1~


    #@jit(nopython=True)                                                                                                                                                    
    def train(self, X_train, Y_train, epoch_num, batch_size=1, cost_fun='regression', lambd=2, weights = None):
        counter = 0
        m = X_train.shape[1]
        batch_numb = math.floor(m/batch_size)
        costs = []
        ind = np.arange(m)

        for epoch in range(1,epoch_num+1):      # from 1 to epoch_num                                                                                                       
            np.random.shuffle(ind)
            X_train = X_train[:,ind]
            Y_train = Y_train[:,ind]
            for j in range(batch_numb):    # go through batches                                                                                                             
                x = X_train[:,j*batch_size:j*batch_size+batch_size]
                y = Y_train[:,j*batch_size:j*batch_size+batch_size]
                y_hat, self.cache = NeuralNet.forward_pass(x,self.params,cost_fun)
                if weights is not None:
                    grads = NeuralNet.back_pass(y, y_hat, cost_fun, self.cache, weights[j*batch_size:j*batch_size+batch_size])
                else:
                    grads = NeuralNet.back_pass(y, y_hat, cost_fun, self.cache, None)
                self.params = NeuralNet.update_params(self.params, grads, lambd)

            if (m % batch_size) != 0:       # the last batch may be of different size                                                                                       
                x = X_train[:,batch_numb*batch_size:]
                y = Y_train[:,batch_numb*batch_size:]
                y_hat, self.cache = NeuralNet.forward_pass(x,self.params,cost_fun)
                if weights is not None:

                    grads = NeuralNet.back_pass(y, y_hat, cost_fun, self.cache, weights[batch_numb*batch_size:])
                else:
                    grads = NeuralNet.back_pass(y, y_hat, cost_fun, self.cache, None)
                self.params = NeuralNet.update_params(self.params, grads, lambd)

            costs.append(NeuralNet.cost(y, y_hat, cost_fun))

            #print("epoch: " + str(epoch) + "   cost: ", NeuralNet.cost(y, y_hat, cost_fun))                                                                                
        return costs

    @staticmethod
    def linear(x,w,b):
        Z = np.dot(w,x)+b
        return Z

    @staticmethod
    def sigmoid(z):
        a = 1/(1+np.exp(-z))
        return a

    @staticmethod
    def relu(z):
        a = np.array(z, copy=True)
        a[a<0]=0
        return a

    @staticmethod
    def linear_back(dZ,w,x):
        m = x.shape[1]
        dx = np.dot(w.T,dZ)
        dw = (1./m)*np.dot(dZ,x.T)
        db = (1./m)*np.sum(dZ, axis = 1, keepdims=True)
        return dx,dw,db

    @staticmethod
    def cannonical_form(params):
        W0,b0,W1,b1 = params
        wh1, wn1 = bounds.nn_wnorm(W0.T)
        nW0 = (wh1.T*128).astype('float').astype('float')/128.
        nb0 = ((b0/wn1.reshape(-1,1))*128).astype('float').astype('float')/128.
        nW1 = (W1*wn1.reshape(1,-1))
        nb1 = b1
        return nW0,nb0,nW1,nb1

    @staticmethod
    def relu_back(da, a):
        dz = np.array(da, copy=True)
        #dz[a>0]=1 is not necessary,  1 * a so de facto a                                                                                                                   
        dz[a<=0]=0
        return dz

    @staticmethod
    def sigmoid_back(da, a):
        dz = da*a*(1-a)
        return dz

    @staticmethod
    def forward_pass(X, params,cost_fun):
        W0,b0,W1,b1,s1 = params
        z1 = NeuralNet.linear(X,W0,b0)
        cache = {}
        cache["X"] = X
        cache["W0"] = W0
        cache["b0"] = b0
        cache["z1"] = z1
        a1 = NeuralNet.relu(z1)
        cache["a1"] = a1
        cache["s1"] = s1
        cache["W1"] = W1
        z2 = NeuralNet.linear(a1,W1,b1)
        cache["z2"] = z2
        if cost_fun=='regression':
            y_hat = z2
        else:
            y_hat = NeuralNet.sigmoid(z2)
        return y_hat, cache

    @staticmethod
    def back_pass(y, y_hat, cost_fun, cache, weights = None):
        m = y.shape[1]
        if cost_fun == 'cross_entropy':
            dy_hat = (1./m)*(-(y/y_hat)+((1-y)/(1-y_hat)) )
            if weights is not None:
                dy_hat = np.multiply(weights,dy_hat)
            dz2 = NeuralNet.sigmoid_back(dy_hat, y_hat)
        if cost_fun == 'quadratic':
            dy_hat = (1./m)*(y_hat - y)
            if weights is not None:
                dy_hat = np.multiply(weights,dy_hat)
            dz2 = NeuralNet.sigmoid_back(dy_hat, y_hat)
        if cost_fun == 'regression':
            dy_hat = (1./m)*(y_hat - y)
            if weights is not None:
                dy_hat = np.multiply(weights,dy_hat)
            dz2=dy_hat
        da1, dw1, db1 = NeuralNet.linear_back(dz2,cache["W1"],cache["a1"])
        dz1 = NeuralNet.relu_back(da1,cache["a1"])
        _, dw0, db0 = NeuralNet.linear_back(dz1,cache["W0"],cache["X"])
        return dw0, db0, dw1, db1

    @staticmethod
    def update_params(params, grads, lambd):
        W0,b0,W1,b1,z1 = params
        dw0, db0, dw1, db1 = grads
        W0 = W0 - lambd*dw0
        b0 = b0 - lambd*db0
        W1 = W1 - lambd*dw1
        b1 = b1 - lambd*db1
        # idea is mamybe z1 can go +,-,0 and set the sign                                                                                                                   
        return W0,b0,W1,b1,z1

    @staticmethod
    def cost(y,y_hat, cost_fun):
        m = y.shape[1]
        if cost_fun == 'cross_entropy':
            cost = -(1./m)*np.sum( np.dot(y,np.log(y_hat).T) + np.dot(1-y,np.log(1-y_hat).T) )
        if cost_fun == 'quadratic':
            cost = (1./(2*m))*np.sum(np.power(y_hat-y,2))
        if cost_fun == 'regression':
            cost = (1./(2*m))*np.sum(np.power(y_hat-y,2))
        return cost

    
