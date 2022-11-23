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
    
    dim = property(lambda x: x.data.ndim)
    shape = property(lambda x: x.data.shape)

    def __add__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, Tensor): 
            out.data = self.data+x.data
        else: 
            out.data = self.data+x
        return out
    
    def __mul__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, Tensor): 
            out.data = self.data*x.data
        else: 
            out.data = self.data*x
        return out

    def __sub__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, Tensor): 
            out.data = self.data-x.data
        else: 
            out.data = self.data-x
        return out

    def __truediv__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, Tensor): 
            out.data = self.data/x.data
        else: 
            out.data = self.data/x
        return out

    def __matmul__(self, x):
        out = Tensor(None, requires_grad=self.requires_grad, backward=self.backward)
        if isinstance(x, Tensor): 
            out.data = self.data@x.data
        else: 
            out.data = self.data@x
        return out
    
    def __getitem__(self, indx):
        return Tensor(self.data[indx])
    
    def tanh(self):
        return Tensor(np.tanh(self.data))
    
    def sum(self, **kwargs):
        return Tensor(self.data.sum(**kwargs))

    def reshape(self, *args):
        return Tensor(self.data.reshape(*args))

    def exp(self):
        return Tensor(np.exp(self.data))

    def log(self):
        return Tensor(np.log(self.data))

    def tolist(self):
        return self.data.tolist()
    