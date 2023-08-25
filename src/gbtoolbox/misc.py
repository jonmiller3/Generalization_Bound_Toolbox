import numpy as np

def grid_to_stack(grid):
    '''stack a d-dimensional grid into an M by d array
    
        Args:
            grid - a grid of points like from the output of meshgrid
        Returns:
            an M by d array, where M is the total number of points that 
            compose the grid, and d is the dimension
    '''
    d = len(grid)
    N = grid[0].size
    xx = [np.reshape(xt,(N,1)) for xt in grid]
    xn = np.hstack(tuple(xx))
    return xn

def gen_stacked_equispaced_nd_grid(N, domain: np.array, indexing='xy'):
    ''' generate a d-dimensional equispaced grid in a stacked form        

        Args:            
            N: number of points per dimension
            domain: domain of points in each dimension, dx2 array each row corresponding to the 
              interval-defined domain for that dimension, where d is the number of dimensions
            indexing: Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.
        Returns:
            N**d x d stacked form of the grid, and unstacked version
    '''    
    x = [np.linspace(dm[0],dm[1],N,endpoint=False) for dm in domain]
    X = np.meshgrid(*x, indexing=indexing) 
    return grid_to_stack(X), x

class ArbitraryPDF:
    '''
    Used to generate random data with an arbitrary PDF
    '''
    def __init__(self, pdf, x):
        '''
            pdf - a numpy array that represents the pdf
            x - the domain of the pdf
        '''
        
        self.x = x
        self.dx = self.x[1]-self.x[0]
        self.cdf = np.cumsum(pdf)*self.dx

    def gen_rv(self,n):
        '''
            Generate n random variables with the PDF
        '''
        u = np.random.random(n)
        inds = np.searchsorted(self.cdf,u)
        return self.x[inds]
