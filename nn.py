import builtins
import math
import random

class Tensor:
    def __init__(self, data, requires_grad=False, backward=None):
        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, (list, int, float)):
            raise TypeError("Cant init with type " + type(data))
        
        self.data = data
        self.requires_grad = requires_grad
        self._backward = backward

        # Compute shape
        self.shape = ()
        curr = data
        while isinstance(curr, (list, Tensor)) and len(curr) != 0:
            self.shape += (len(curr), )
            curr = curr[0]
        if self.dim > 2: 
            raise Exception("Dimensions over 2 not implemented")

        # For 2d tensors, convert the nested arrays to tensors
        if self.dim == 2 and self.shape[1] > 0 and isinstance(self.data[0], list):
            self.data = [Tensor(vec, requires_grad=self.requires_grad) for vec in self.data]
        
        if self.requires_grad and self.dim == 1 and self.shape[0] > 0:
            self._grad = Tensor([0] * self.shape[0])

    def get_grad(self):
        if self.dim == 1:
            return self._grad
        if self.dim == 2:
            return Tensor([vec.grad for vec in self.data])
    
    def set_grad(self, x):
        if isinstance(x, (int, float)):
            if self.dim == 1:
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
    
    def __mul__(self, x):
        if isinstance(x, (int, float)):
            if self.dim == 1:
                return Tensor([a*x for a in self.data])
            if self.dim == 2:
                return Tensor([vec*x for vec in self.data])

        dims = (self.dim, x.dim)
        if dims == (2,1): return x*self

        if dims == (1,1):
            if self.shape != x.shape: 
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')

            data = [a*b for a, b in zip(self.data, x.data)]
            def backward():
                self.grad += x * out.grad
                x.grad += self * out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=backward)
            return out

        if dims == (1,2): 
            return Tensor([self*v for v in x.data])

        if dims == (2,2): 
            return Tensor([a*b for a,b in zip(self.data, x.data)])

    def __add__(self, x):
        if isinstance(x, (int, float)):
            if self.dim == 1:
                return Tensor([a+x for a in self.data])
            if self.dim == 2:
                data = [vec+x for vec in self.data]
                return Tensor(data)
        
        dims = (self.dim, x.dim)
        if dims == (2,1): return x+self

        if dims == (1,1): 
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')

            data = [a+b for a, b in zip(self.data, x.data)]
            def back():
                self.grad += out.grad
                x.grad += out.grad
            out = Tensor(data, requires_grad=self.requires_grad, backward=back)
            return out

        if dims == (1,2):
            return Tensor([self+v for v in x.data])
        
        if dims == (2,2):
            return Tensor([a+b for a,b in zip(self.data, x.data)])
    
    def __sub__(self, x):
        if isinstance(x, (int, float)):
            if self.dim == 1:
                return Tensor([a-x for a in self.data])
            if self.dim == 2:
                return Tensor([vec-x for vec in self.data])
        
        dims = (self.dim, x.dim)

        if dims == (1,1):
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')
            return Tensor([a-b for a, b in zip(self.data, x.data)])

        if dims == (1,2):
            return Tensor([self-v for v in x.data])

        if dims == (2,1):
            return Tensor([v-x for v in self.data])
    
    def __truediv__(self, x):
        if isinstance(x, (int, float)):
            if self.dim == 1:
                return Tensor([a/x for a in self.data])
            if self.dim == 2:
                return Tensor([vec/x for vec in self.data])
        
        dims = (self.dim, x.dim)

        if dims == (1,1):
            if self.shape != x.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {x.shape}')
            
            out = [a/b for a, b in zip(self.data, x.data)]

            def a_div_b_ddb(a, b):   return -a * math.pow(b, -2) # d/dy x/y
            def back():
                self.grad += Tensor([1/a for a in x.data])
                x.grad += Tensor([a_div_b_ddb(a, b) for a,b in zip(self.data, x.data)])
            return Tensor(out, backward=back)

        if dims == (2,1):
            return Tensor([vec/x for vec in self.data])

        if dims == (1,2):
            return Tensor([self/vec for vec in x.data])

        if dims == (2,2):
            return Tensor([a/b for a,b in zip(self.data, x.data)])
        
    def __matmul__(self, other):
        dims = (self.dim, other.dim)
        
        if dims == (1, 1):
            if self.shape != other.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

            prod = 0
            for a, b in zip(self.data, other.data): prod += a*b
            return Tensor(prod)

        if dims == (1, 2):
            if self.shape[0] != other.shape[0]:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')
            
            data = Tensor([0] * other.shape[1])
            for i,x in enumerate(self.data):
                intermediate = Tensor([x*w for w in other.data[i]])
                data = data + intermediate
            return data

        if dims == (2,1) or dims == (2,2):
            return Tensor([(v@other).data for v in self.data])
    
    def log(self):
        if self.dim == 1:
            return Tensor([math.log(x) if x > 0 else math.nan for x in self.data])
        if self.dim == 2:
            return Tensor([v.log() for v in self.data])
    
    def exp(self):
        if self.dim == 1:
            return Tensor([math.exp(x) for x in self.data])
        if self.dim == 2:
            return Tensor([v.exp() for v in self.data])
    
    def __len__(self):
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

    def tolist(self):
        if self.dim == 1:
            return self.data
        if self.dim == 2:
            return [t.data for t in self.data]

def tanh(tensor):
    if tensor.dim == 1:
        return Tensor([math.tanh(x) for x in tensor.data])
    if tensor.dim == 2:
        return Tensor([tanh(t) for t in tensor.data])

def ones(shape):
    if not isinstance(shape, tuple):
        raise TypeError("Expected tuple")

    dim = len(shape)
    if dim == 0 or dim > 2:
        raise RuntimeError("Invalid size")

    if dim == 1:
        data = [1] * shape[0]
        return Tensor(data)
    if dim == 2:
        data = []
        for _ in range(shape[0]):
            data.append([1] * shape[1])
        return Tensor(data)

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
        data = [sum(t) for t in tensor.data]
        return Tensor(data)

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