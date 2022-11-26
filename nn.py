import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, backward=None):
        if data is None:
            raise Exception("data none")
        if isinstance(data, list):
            data = np.array(data, dtype=float)
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=float)
        
        self.data = data
        self.requires_grad = requires_grad
        self.backward = backward

        if not requires_grad: return
        self._grad = np.zeros(self.data.shape)
    

    def set_grad(self, x):
        if isinstance(x, (int, float)):
            self._grad = np.full(self.shape, x, dtype=float)
        else:
            self._grad = x

    def get_grad(self):
        return self._grad 
    
    grad = property(get_grad, set_grad)
    dim = property(lambda x: x.data.ndim)
    shape = property(lambda x: x.data.shape)

    def __add__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        out_d = self.data+data

        def back():
            self_dim = self.dim if self.shape != (1,) else 0
            x_dim = data.ndim if data.shape != (1,) else 0
            dims = (self_dim, x_dim)

            if dims in [(0,1), (0,2)]:
                self.grad += np.sum(out.grad)
                x.grad += out.grad
            if dims in [(1,0), (2,0)]:
                self.grad += out.grad
                x.grad += np.sum(out.grad)
            if dims in [(0,0), (1,1), (2,2)]:
                self.grad += out.grad
                x.grad += out.grad
            if dims in [(1,2)]:
                self.grad += np.sum(out.grad, axis=0)
                x.grad += out.grad
            if dims in [(2,1)]:
                self.grad += out.grad
                x.grad += np.sum(out.grad, axis=0)

        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back)
        return out
    
    def __mul__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        return Tensor(self.data*data, requires_grad=self.requires_grad, backward=self.backward)
    

    def __sub__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        return Tensor(self.data-data, requires_grad=self.requires_grad, backward=self.backward)

    def __truediv__(self, x):
        data = x.data if isinstance(x, Tensor) else x
        return Tensor(self.data/data, requires_grad=self.requires_grad, backward=self.backward)

    def __matmul__(self, x):
        data = x.data if isinstance(x, Tensor) else x

        out_d = self.data@data
        def back():
            dims = (self.dim, data.ndim)
            if dims == (1,2):
                self.grad += data.sum(axis=1)
                x.grad += (self.data.reshape((-1, 1)) * np.ones(x.shape))
        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back)
        return out
    
    def __rmul__(self, x):
        return self*x
    
    def __getitem__(self, *args):
        return Tensor(self.data.__getitem__(*args))
    
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
    
    def __repr__(self):
        repr = self.data.__repr__()
        if "array" in repr:
            repr = "tensor" + repr[5:]
        return repr
    