import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, backward=None):
        if isinstance(data, list):
            data = np.array(data, dtype=float)
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=float)
        
        self.data = data
        self.requires_grad = requires_grad
        self.backward = backward

    def __add__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, np.ndarray): 
            out.data = self.data+x
        else: 
            out.data = self.data+x.data
        return out
    
    def __mul__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, np.ndarray): 
            out.data = self.data*x
        else: 
            out.data = self.data*x.data
        return out

    def __sub__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, np.ndarray): 
            out.data = self.data-x
        else: 
            out.data = self.data-x.data
        return out

    def __truediv__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, np.ndarray): 
            out.data = self.data/x
        else: 
            out.data = self.data/x.data
        return out

    def __matmul__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, np.ndarray): 
            out.data = self.data@x
        else: 
            out.data = self.data@x.data
        return out
    
    def tolist(self):
        return self.data.tolist()
    
    dim = property(lambda x: x.data.ndim)
    shape = property(lambda x: x.data.shape)