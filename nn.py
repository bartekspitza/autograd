import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, backward=None):
        if not isinstance(data, (list, int, float)):
            raise TypeError("Cant init with type " + type(data))
        
        self.data = data
        self.requires_grad = requires_grad
        self._backward = backward