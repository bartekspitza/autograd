from .tensor import Tensor
import numpy as np


class MLP:
    def __init__(self, inputs=1, hidden=(1,), outs=1):
        self.W = []
        self.b = []
        self.rng = np.random.default_rng(seed=5)

        for i, layer in enumerate(hidden):
            prev = inputs if i == 0 else hidden[i-1]

            W = Tensor(self.rng.normal(size=(prev, layer)))
            self.W.append(W)
            b = Tensor(self.rng.normal(size=(layer,)))
            self.b.append(b)

        out = Tensor(self.rng.normal(size=(hidden[-1], outs)))
        self.W.append(out)
        self.parameters = self.W + self.b
    
    def forward(self, x):
        # Layers
        for w,b in zip(self.W, self.b):
            x = b + (x@w)
            x = x.tanh()
        
        # Output layer
        x = x@self.W[-1]

        # Softmax
        return softmax(x)
    
    def train(self, lr=0.01):
        for p in self.parameters:
            p.data -= lr * p.grad
        
        self.zero_grad()
    
    def zero_grad(self):
        for p in self.parameters: 
            p.grad = 0 
    
    def __call__(self, x):
        return self.forward(x)

def softmax(x):
    x = x.exp()
    if x.dim == 1:
        return x / x.sum()
    elif x.dim == 2:
        return x / x.sum(axis=x.dim-1).reshape((-1, 1))