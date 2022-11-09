import builtins
import math

class Tensor:
    def __init__(self, data):
        if not isinstance(data, list):
            raise TypeError("Expected python list")

        self.data = data
        self.shape = ()
        curr_dim = data
        while True:
            if isinstance(curr_dim, (list, Tensor)):
                curr_dim_len = len(curr_dim) if isinstance(curr_dim, list) else len(curr_dim.data)
                self.shape = self.shape + (curr_dim_len, )

                if curr_dim_len > 0:
                    curr_dim = curr_dim[0]
                else:
                    break
            else:
                break
        self.dim = len(self.shape)

        if self.dim == 1:
            return
        if self.dim > 2:
            raise Exception("Dimensions over 2 not implemented")

        # For 2d tensors, convert the nested arrays to tensors
        # Watchout for already "good" tensors
        if self.shape[1] > 0 and isinstance(self.data[0], Tensor):
            return
        self.data = [Tensor(vec) for vec in self.data]
    
    def mult(self, other):
        if isinstance(other, (int, float)):
            if self.dim == 1:
                return Tensor([a*other for a in self.data])
            if self.dim == 2:
                data = [[a*other for a in vec.data] for vec in self.data]
                return Tensor(data)
        
        if not isinstance(other, Tensor):
            raise TypeError("Not Tensor")
        
        # v*v
        if self.dim == 1 and other.dim == 1:
            if self.shape != other.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

            new_data = [a*b for a, b in zip(self.data, other.data)]	
            return Tensor(new_data)
        # v*m
        if self.dim == 1 and other.dim == 2:
            return other*self;
        # m*v
        if self.dim == 2 and other.dim == 1:
            data = [v*other for v in self.data]
            return Tensor(data)
    
    def add(self, other):
        if isinstance(other, (int, float)):
            if self.dim == 1:
                return Tensor([a+other for a in self.data])
            if self.dim == 2:
                data = [[a+other for a in vec.data] for vec in self.data]
                return Tensor(data)
        
        if not isinstance(other, Tensor):
            raise TypeError("Not Tensor")
        
        # v + v
        if self.dim == 1 and other.dim == 1:
            if self.shape != other.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

            data = [a+b for a, b in zip(self.data, other.data)]	
            return Tensor(data)
        # v + m
        if self.dim == 1 and other.dim == 2:
            return other+self
        # m + v
        if self.dim == 2 and other.dim == 1:
            data = [v+other for v in self.data]
            return Tensor(data)
    
    def sub(self, other):
        if isinstance(other, (int, float)):
            if self.dim == 1:
                return Tensor([a-other for a in self.data])
            if self.dim == 2:
                data = [[a-other for a in vec.data] for vec in self.data]
                return Tensor(data)
        
        if not isinstance(other, Tensor):
            raise TypeError("Not Tensor")
        
        # v - v
        if self.dim == 1 and other.dim == 1:
            if self.shape != other.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

            new_data = [a-b for a, b in zip(self.data, other.data)]	
            return Tensor(new_data)
        # v - m
        if self.dim == 1 and other.dim == 2:
            data = [self-v for v in other.data]
            return Tensor(data)
        # m - v
        if self.dim == 2 and other.dim == 1:
            data = [v-other for v in self.data]
            return Tensor(data)
    
    def div(self, other):
        if isinstance(other, (int, float)):
            if self.dim == 1:
                return Tensor([a/other for a in self.data])
            if self.dim == 2:
                data = [[a/other for a in vec.data] for vec in self.data]
                return Tensor(data)

        if not isinstance(other, Tensor):
            raise TypeError("Not Tensor")
        
        # v / v
        if self.dim == 1 and other.dim == 1:
            if self.shape != other.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

            new_data = [a/b for a, b in zip(self.data, other.data)]	
            return Tensor(new_data)
        # m / v
        if self.dim == 2 and other.dim == 1:
            data = [vec/other for vec in self.data]
            return Tensor(data)
        # v / m
        if self.dim == 1 and other.dim == 2:
            data = [self/vec for vec in other.data]
            return Tensor(data)
        
    def dot(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Not Tensor")
        
        # v dot v
        if self.dim == 1 and other.dim == 1:
            if self.shape != other.shape:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

            prod = 0
            for a, b in zip(self.data, other.data):
                prod += a*b
            return prod
        # v dot m
        if self.dim == 1 and other.dim == 2:
            if self.shape[0] != other.shape[0]:
                raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')
            
            data = Tensor([0] * other.shape[1])
            for i,x in enumerate(self.data):
                intermediate = Tensor([x*w for w in other.data[i]])
                data = data + intermediate
            return data
        # m dot (v or m)
        if self.dim == 2 and (other.dim == 1 or other.dim == 2):
            data = [v@other for v in self.data]
            return Tensor(data)
    
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
        
    def __getitem__(self, indx):
        if isinstance(indx, int):
            return self.data[indx]
        raise RuntimeError("Not implemented")
    
    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mult(other)

    def __truediv__(self, other):
        return self.div(other)
        
    def __matmul__(self, other):
        return self.dot(other)
        
    def __repr__(self):
        return f'Tensor(data={self.data.__repr__()})'
    
    # Quality of life
    def tolist(self):
        if self.dim == 1:
            return self.data
        if self.dim == 2:
            return [t.data for t in self.data]

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

def sum(tensor):
    if not isinstance(tensor, Tensor):
        raise TypeError("Expected tensor")

    if tensor.dim == 1:
        return builtins.sum(tensor.data)
    if tensor.dim == 2:
        data = [sum(t) for t in tensor.data]
        return Tensor(data)