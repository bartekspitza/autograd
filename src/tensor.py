import numpy as np

# Readability,S=Scalar, V=Vector, M=Matrix, e.g. SS means scalar to scalar
SS,VV,MM,SV,SM,VS,MS,VM,MV = (0,0),(1,1),(2,2),(0,1),(0,2),(1,0),(2,0),(1,2),(2,1)

class Tensor:
    def __init__(self, data, requires_grad=False, backward=None, children=()):
        if data is None:
            raise Exception("data none")
        if isinstance(data, list):
            data = np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            data = data.astype(float)
        elif isinstance(data, (int, float)):
            data = np.array([data], dtype=float)
        
        self.data = data
        self.requires_grad = requires_grad
        self._backward = backward
        self._prev = children

        #if not requires_grad: return
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

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if self.grad is None or self.grad.sum() == 0: 
            self.grad = 1

        for v in reversed(topo):
            if v._backward is not None:
                v._backward()

    ## Convienence method to get a tuple of two tensors' dims
    def _dims(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Expected ndarray")
        return ((self.dim if self.shape != (1,) else 0), (x.ndim if x.shape != (1,) else 0))

    def __add__(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x_data = x.data if isinstance(x, Tensor) else x

        def back():
            dims = self._dims(x_data)

            if dims in [SS, VV, MM]:
                self.grad += out.grad
                x.grad += out.grad
            if dims in [SV, SM]:
                self.grad += np.sum(out.grad)
                x.grad += out.grad
            if dims in [VS, MS]:
                self.grad += out.grad
                x.grad += np.sum(out.grad)
            if dims == VM:
                self.grad += np.sum(out.grad, axis=0)
                x.grad += out.grad
            if dims == MV:
                self.grad += out.grad
                x.grad += np.sum(out.grad, axis=0)

        out = Tensor(self.data+x_data, requires_grad=self.requires_grad, backward=back, children=(self, x))
        return out
    
    def __mul__(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x_data = x.data if isinstance(x, Tensor) else x

        def back(): 
            dims = self._dims(x_data)

            if dims in [SS, VV, MM]:
                self.grad += x_data * out.grad
                x.grad += self.data * out.grad
            if dims in [SV, SM]:
                self.grad += (x_data * out.grad).sum()
                x.grad += self.data * out.grad
            if dims in [VS, MS]:
                self.grad += x_data * out.grad
                x.grad += (self.data * out.grad).sum()
            if dims == VM:
                self.grad += (x_data * out.grad).sum(axis=0)
                x.grad += self.data * out.grad
            if dims == MV:
                self.grad += x_data * out.grad
                x.grad += (self.data * out.grad).sum(axis=0)

        out = Tensor(self.data*x_data, requires_grad=self.requires_grad, backward=back, children=(self, x))
        return out

    def __sub__(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x_data = x.data if isinstance(x, Tensor) else x

        def back():
            dims = self._dims(x_data)
            if dims in [SS, VV, MM]:
                self.grad += out.grad
                x.grad -= out.grad
            if dims in [SV, SM]:
                self.grad += out.grad.sum()
                x.grad -= out.grad
            if dims in [VS, MS]:
                self.grad += out.grad
                x.grad -= out.grad.sum()
            if dims == VM:
                self.grad += out.grad.sum(axis=0)
                x.grad -= out.grad
            if dims == MV:
                self.grad += out.grad
                x.grad -= out.grad.sum(axis=0)

        out = Tensor(self.data-x_data, requires_grad=self.requires_grad, backward=back, children=(self, x))
        return out

    def __truediv__(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x_data = x.data if isinstance(x, Tensor) else x

        out_d = self.data/x_data
        def back():
            dims = self._dims(x_data)

            denom_ddx = -self.data * np.power(x_data, -2) # d/dx y/x = -yx^-2

            s_grad = out.grad/x_data
            x_grad = denom_ddx*out.grad

            if dims in [SV, SM]: s_grad = s_grad.sum()
            if dims in [VS, MS]: x_grad = x_grad.sum()
            if dims == VM: s_grad = s_grad.sum(axis=0)
            if dims == MV: x_grad = x_grad.sum(axis=0)

            if dims == MM:
                if self.shape != x.shape:
                    if self.shape[1] == 1:
                        self.grad += s_grad.sum(axis=1).reshape(-1, 1)
                        x.grad += x_grad
                    elif x.shape[1] == 1:
                        self.grad += s_grad
                        x.grad += x_grad.sum(axis=1).reshape(-1, 1)
                else:
                    self.grad += s_grad
                    x.grad += x_grad
            else:
                self.grad += s_grad
                x.grad += x_grad

        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back, children=(self, x))
        return out

    def __matmul__(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x_data = x.data if isinstance(x, Tensor) else x

        def back():
            dims = self._dims(x_data)

            if dims == VV:
                self.grad += x_data * out.grad
                x.grad += self.data * out.grad
            if dims == MV:
                out_g_rs = out.grad.reshape(-1,1) * np.ones((self.shape)) # e.g. turns [3, 4] -> [[3, 3], [4, 4]]
                self.grad += x_data * out_g_rs
                x.grad += (self.data * out_g_rs).sum(axis=0)
            if dims == VM:
                out_g_rs = np.tile(out.grad, (len(x_data), 1))              # e.g. turns [3, 4] -> [[3, 4], [3, 4]]
                out_g_rs_T = out.grad.reshape(-1,1) * np.ones((self.shape)) # e.g. turns [3, 4] -> [[3, 3], [4, 4]]
                self.grad += (x_data.T * out_g_rs_T).sum(axis=0)
                x.grad += self.data.reshape(-1,1) * out_g_rs
            if dims == MM:
                # left matrix has shape (k,m)
                # right matrix has shape (m, n)
                k,m,n = len(self.data), x_data.shape[0], x_data.shape[1]
                grads = out.grad.repeat(m, axis=0).reshape((k,m,n))
                self.grad += (x_data*grads).sum(axis=2)
                x.grad += (self.data.reshape((k,m,1))*grads).sum(axis=0)


        out = Tensor(self.data@x_data, requires_grad=self.requires_grad, backward=back, children=(self, x))
        return out
    
    def __rmul__(self, x):
        return self*x
    
    def __len__(self):
        return len(self.data)
    
    def __neg__(self):
        return self*-1
    
    def __getitem__(self, *args):
        return Tensor(self.data.__getitem__(*args))
    
    def tanh(self):
        out_d = np.tanh(self.data)
        def back():
            self.grad += (1 - np.power(out_d, 2)) * out.grad

        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back, children=(self,))
        return out

    def relu(self):
        tmp = self.data >= 0
        out_d = tmp*self.data

        def back():
            self.grad += tmp * out.grad

        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back, children=(self,))
        return out
    
    def sum(self, **kwargs):
        out_d = self.data.sum(**kwargs)
        def back():
            if 'axis' in kwargs:
                axis = kwargs['axis']
                if axis == 0:
                    self.grad += np.tile(out.grad, len(self.data)).reshape(self.shape)
                elif axis == 1:
                    self.grad += np.ones(self.shape) * out.grad.reshape(-1,1)
            else:
                self.grad += np.full(self.shape, out.grad)

        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back, children=(self,))
        return out

    def reshape(self, *args):
        def back():
            self.grad = out.grad.reshape(*args)
        out = Tensor(self.data.reshape(*args), backward=back, children=[self])
        return out

    def exp(self):
        out_d = np.exp(self.data)
        def back(): self.grad += out_d * out.grad
        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back, children=(self,))
        return out

    def log(self):
        out_d = np.log(self.data)
        def back(): self.grad += out.grad/self.data
        out = Tensor(out_d, requires_grad=self.requires_grad, backward=back, children=(self,))
        return out
    
    # Numpy wrappers
    def std(self, *args, **kwargs): return self.data.std(*args, **kwargs)
    def mean(self, *args, **kwargs): return self.data.mean(*args, **kwargs)
    def var(self, *args, **kwargs): return self.data.var(*args, **kwargs)
    def tolist(self): return self.data.tolist()
    
    def __repr__(self):
        repr = self.data.__repr__()
        if "array" in repr:
            repr = "tensor" + repr[5:]
        return repr
    