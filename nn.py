import builtins
import math
import random

class Tensor:

    def __init__(self, data, requires_grad=False, backward=None):
        if not isinstance(data, (list, int, float)):
            raise TypeError("Cant init with type " + type(data))
        
        self.data = data
        self.requires_grad = requires_grad
        self._backward = backward

        if isinstance(data, list):
            self.shape = (len(data),)
            if self.shape != (0,): 
                if not isinstance(data[0], Tensor):
                    raise TypeError("Found nested non-tensor item")
                self.shape += data[0].shape
        else:
            self.shape = ()
        
        if not self.requires_grad:
            return

        if self.dim == 0:
            self._grad = Tensor(0)
        if self.dim == 1 and self.shape[0] > 0:
            self._grad = Tensor([0] * self.shape[0])

    def get_grad(self):
        if self.dim < 2:
            return self._grad
        else:
            return Tensor([vec.grad for vec in self.data])
    
    def set_grad(self, x):
        if isinstance(x, (int, float)):
            if self.dim == 0:
                self._grad = Tensor(x)
            elif self.dim == 1:
                self._grad = Tensor([x] * len(self))
            elif self.dim == 2:
                for vec in self.data:
                    vec.grad = x
        else:
            if self.shape != x.shape:
                raise RuntimeError(f"Cant set gradients of {x.shape=} to tensor of {self.shape=}")
            self._grad = x
    
    def get_backward(self):
        if self.dim == 2:
            def back():
                for v in self.data:
                    v.backward()
            return back

        return self._backward
    
    dim = property(lambda x: len(x.shape))
    grad = property(get_grad, set_grad)
    backward = property(get_backward)

    def __add__(self, x):
        if isinstance(x, (int, float)):
            x = Tensor(x)

        if self.dim < x.dim: return x+self

        dims = (self.dim, x.dim)

        if dims == (0,0):
            data = self.data+x.data
            def back():
                self.grad += out.grad
                x.grad += out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=back)
            return out

        if dims == (1,0):
            return Tensor([a+x for a in self.data])
        if dims == (2,0):
            data = [vec+x for vec in self.data]
            return Tensor(data)

        if dims == (1,1): 
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')

            data = [a+b for a, b in zip(self.data, x.data)]
            def back():
                self.grad += out.grad
                x.grad += out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=back)
            return out

        if dims == (2,1):
            return Tensor([v+x for v in self.data])
        if dims == (2,2):
            return Tensor([a+b for a,b in zip(self.data, x.data)])
    
    def __mul__(self, x):
        if isinstance(x, (int, float)):
            x = Tensor(x)

        if self.dim < x.dim: return x*self

        dims = (self.dim, x.dim)

        if dims == (0,0): 
            data = self.data*x.data
            def back():
                self.grad += x*out.grad
                x.grad += self*out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=back)
            return out

        if dims == (1,0):
            prod = []
            summ = 0
            for a in self.data:
                summ += a
                prod.append(a*x)

            def back():
                self.grad += x*out.grad
                x.grad += sum(out.grad*summ)

            out = Tensor(prod, backward=back)
            return out
        if dims == (2,0):
            return Tensor([vec*x for vec in self.data])

        if dims == (1,1):
            if self.shape != x.shape: 
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')

            data = [a*b for a, b in zip(self.data, x.data)]
            def backward():
                self.grad += x * out.grad
                x.grad += self * out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=backward)
            return out

        if dims == (2,1): 
            return Tensor([x*v for v in self.data])
        if dims == (2,2): 
            return Tensor([a*b for a,b in zip(self.data, x.data)])
    
    def __sub__(self, x):
        if isinstance(x, (int, float)):
            x = Tensor(x)
        
        dims = (self.dim, x.dim)
        if dims == (0, 0): return Tensor(self.data-x.data)
        if dims == (1, 0): return Tensor([a-x for a in self.data])
        if dims == (0, 1): return Tensor([self-a for a in x.data])
        if dims == (2, 0): return Tensor([a-x for a in self.data])
        if dims == (0, 2): return Tensor([self-a for a in x.data])

        if dims == (1,1):
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')

            data = [a-b for a, b in zip(self.data, x.data)]
            def back():
                self.grad += out.grad
                x.grad -= out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=back)
            return out

        if dims == (1,2):
            return Tensor([self-v for v in x.data])
        if dims == (2,1):
            return Tensor([v-x for v in self.data])
        if dims == (2,2):
            return Tensor([a-b for a,b in zip(self.data, x.data)])
    
    def __truediv__(self, x):
        if isinstance(x, (int, float)):
            x = Tensor(x)
        
        dims = (self.dim, x.dim)
        if dims == (0, 0): return Tensor(self.data/x.data)
        if dims == (1, 0): return Tensor([a/x for a in self.data])
        if dims == (0, 1): return Tensor([self/a for a in x.data])
        if dims == (2, 0): return Tensor([a/x for a in self.data])
        if dims == (0, 2): return Tensor([self/a for a in x.data])

        if dims == (1,1):
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')
            
            out = [a/b for a, b in zip(self.data, x.data)]

            def back():
                self.grad += Tensor([1/a for a in x.data]) * out.grad
                x.grad += Tensor([-a*math.pow(b, -2) for a,b in zip(self.data, x.data)]) * out.grad
            out = Tensor(out, backward=back)
            return out

        if dims == (2,1):
            return Tensor([vec/x for vec in self.data])
        if dims == (1,2):
            return Tensor([self/vec for vec in x.data])
        if dims == (2,2):
            return Tensor([a/b for a,b in zip(self.data, x.data)])
        
    def __matmul__(self, x):
        dims = (self.dim, x.dim)
        
        if dims == (1, 1):
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')

            out = 0
            for a,b in zip(self.data, x.data): out += a*b

            def back():
                self.grad += x.data * out.grad
                self.grad += self.data * out.grad
            out._backward = back
            return out

        if dims == (1, 2):
            if self.shape[0] != x.shape[0]:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')
            
            data = zeros((x.shape[1],))
            for i,vec in enumerate(self.data):
                intermediate = Tensor([vec*w for w in x.data[i]])
                data = data + intermediate
            return data

        if dims == (2,1) or dims == (2,2):
            return Tensor([(v@x) for v in self.data])
    
    def __radd__(self, x):
        return self+x

    def __rmult__(self, x):
        return self*x
    
    def log(self):
        if self.dim == 0:
            return Tensor(math.log(self.data) if self.data > 0 else math.nan)
        if self.dim == 1:
            return Tensor([x.log() for x in self.data])
        if self.dim == 2:
            return Tensor([v.log() for v in self.data])
    
    def exp(self):
        if self.dim == 0:
            return Tensor(math.exp(self.data))
        if self.dim == 1:
            return Tensor([x.exp() for x in self.data])
        if self.dim == 2:
            return Tensor([v.exp() for v in self.data])
    
    def __len__(self):
        if self.dim == 0:
            return 0
        return len(self.data)
        
    def data_repr(self):
        if self.dim == 0:
            return f'({self.data})'
        if self.dim == 1:
            return self.data.__repr__()
        if self.dim == 2:
            repr = "["
            for i, vec in enumerate(self.data):
                if i != 0: repr += " "
                repr += vec.data_repr()
                if i != len(self.data)-1: repr += ",\n"
            repr += "]"
            return repr

    def __repr__(self):
        return f'Tensor(data={self.data.__repr__()})'
    
    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise RuntimeError("Expected int")

        self.data[key] = value

    def __getitem__(self, indx):
        if isinstance(indx, int):
            return self.data[indx]
        if isinstance(indx, list):
            data = [self[num] for num in indx]
            return Tensor(data)

        raise RuntimeError("Not implemented")

    def unwrap(self):
        if self.dim == 0: return self.data
        return [x.unwrap() for x in self.data]

def tanh(tensor):
    if tensor.dim == 0:
        return Tensor(math.tanh(tensor.data))
    if tensor.dim == 1:
        return Tensor([tanh(x) for x in tensor.data])
    if tensor.dim == 2:
        return Tensor([tanh(t) for t in tensor.data])

def tensor_of_number(shape, num):
    if not isinstance(shape, tuple):
        raise TypeError("Expected tuple")

    dim = len(shape)
    if dim == 0 or dim > 2:
        raise RuntimeError("Invalid size")

    if dim == 1:
        return wrap([num] * shape[0])
    if dim == 2:
        data = [([num] * shape[1]) for _ in range(shape[0])]
        return wrap(data)

def ones(shape):
    return tensor_of_number(shape, 1)

def zeros(shape):
    return tensor_of_number(shape, 0)

def randn(shape):
    if not isinstance(shape, tuple):
        raise TypeError("Expected tuple")
    
    dim = len(shape)
    if dim == 0 or dim > 2:
        raise RuntimeError("Invalid size")

    if dim == 1:
        data = [random.gauss(mu=0.0, sigma=1.0) for _ in range(shape[0])]
        return Tensor(data)
    if dim == 2:
        data = [randn(shape[1:]) for _ in range(shape[0])]
        return Tensor(data)

def sum(tensor):
    if not isinstance(tensor, Tensor):
        raise TypeError("Expected tensor")

    if tensor.dim == 1:
        return builtins.sum(tensor.data)
    if tensor.dim == 2:
        sm = 0
        for v in tensor.data:
            sm += sum(v)
        return Tensor(sm.data)

def multinomial(input, num_samples, replacement=True, indices=None):
    if input.dim != 2:
        raise RuntimeError("Only dim=2 supported")
    
    new_shape = list(input.shape)
    new_shape[0] = num_samples
    out = ones(tuple(new_shape))
    end = len(input)
    for i in range(num_samples):
        x = random.randrange(0, end)
        out[i] = input[x]

        if not indices is None:
            indices.append(x)
    
    return out

def wrap(data):
    if isinstance(data, (int, float)):
        return Tensor(data)

    elif isinstance(data, list):
        wrapped = []
        prev_shape = None

        for x in data:
            item = wrap(x)
            if not prev_shape is None and item.shape != prev_shape:
                raise RuntimeError("found inconsistent vector sizes")
            
            prev_shape = item.shape
            wrapped.append(item)
        
        return Tensor(wrapped)

    raise TypeError(f"Unexpected type={type(data)}")