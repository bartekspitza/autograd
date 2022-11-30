from .tensor import Tensor
import numpy as np

rng = np.random.default_rng(seed=5)

class Linear:
    def __init__(self, ins, outs, bias=True):
        self.W = Tensor(rng.normal(size=(ins, outs)))
        self.b = None
        if bias:
            self.b = Tensor(rng.normal(size=(outs,)))
    
    def parameters(self):
        return [self.W] if self.b is None else [self.W, self.b]
    
    def __call__(self, x):
        if self.b:
            return x@self.W + self.b
        else:
            return x@self.W

class BatchNorm:

    def __init__(self):pass

    def __call__(self, x):pass


class Tanh:
    def __call__(self, x):
        return x.tanh()

class Softmax:
    def __call__(self, x):
        return softmax(x)


class MLP:

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
            if i == breakpoint: return x
            if printstddev and isinstance(layer, Linear): print(f'std={x.std()}')
        
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            if isinstance(l, Linear):
                params += l.parameters()
        return params
    
    def optimize(self, lr=0.01):
        for p in self.parameters():
            p.data -= lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0 
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

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

def softmax(x):
    x = x.exp()
    if x.dim == 1:
        return x / x.sum()
    elif x.dim == 2:
        return x / x.sum(axis=x.dim-1).reshape((-1, 1))