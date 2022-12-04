from .tensor import Tensor
import numpy as np
from operator import attrgetter

rng = np.random.default_rng(seed=5)

class Linear:
    def __init__(self, ins, outs, bias=True):
        self.W = Tensor(rng.normal(size=(ins, outs)))
        self.bias = Tensor(np.zeros((outs,))) if bias else None
    
    def parameters(self):
        return [self.W, self.bias] if self.bias else [self.W]
    
    def __call__(self, x):
        return (x@self.W + self.bias) if self.bias else (x@self.W)

class BatchNorm:
    def __init__(self, outs, eps=1e-5, momentum=0.1):
        self.beta = Tensor(np.zeros((outs,)))
        self.gamma = Tensor(np.ones((outs,)))
        self.eps = eps
        self.momentum = momentum

        self.training = True
        self._count = 0
        self.mean_running = np.zeros((outs,))
        self.var_running = np.zeros((outs,))

    def __call__(self, x):
        if self.training:
            x_var = x.var(0)
            x_mean = x.mean(0)
            self.mean_running = (1-self.momentum) * self.mean_running + (self.momentum*x_mean)
            self.var_running = (1-self.momentum) * self.var_running + (self.momentum*x_var)
        else:
            x_var = self.var_running
            x_mean = self.mean_running

        x = self.gamma * ((x-x_mean)/(np.sqrt(x_var) +self.eps)) + self.beta
        return x

    def parameters(self):
        return [self.beta, self.gamma]

class Tanh:
    def __call__(self, x):
        return x.tanh()

class Softmax:
    def __call__(self, x):
        x = x.exp()
        if x.dim == 1:
            return x / x.sum()
        elif x.dim == 2:
            return x / x.sum(axis=x.dim-1).reshape((-1, 1))


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, printstddev=False, breakpoint=-1):
        """
        Forwards the input through the network.
        printstddev - will print the std.dev before act-function of each layer
        breakpoint - a tuple: breakpoint[0] = index of layer, breakpoint[1] = 0 for pre act, 1 for after act. 
        Then returns that layers output
        """

        # Layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if printstddev and isinstance(layer, (Linear, BatchNorm)): print(f'std={x.std()}')
            if i == breakpoint: return x
        
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            if isinstance(l, (Linear, BatchNorm)):
                params += l.parameters()
        return params
    
    def inference(self):
        for l in (l for l in self.layers if isinstance(l, BatchNorm)):
            l.training=False
    
    def optimize(self, lr=0.01):
        for p in self.parameters():
            p.data -= lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0 
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def __getitem__(self, x):
        return self.layers[x]

def nlll(x, target, reduction=None):
    """
    The negative likelihood log loss.
    x - the probability distribution (softmax)
    y - the target probability distribution
    """
    assert x.shape == target.shape

    nll = -(x.log() * target).sum(axis=x.dim-1)

    if x.dim == 2:
        if reduction == 'sum':
            return nll.sum()
        if reduction == 'mean':
            return nll.sum() / len(nll) 

    return nll