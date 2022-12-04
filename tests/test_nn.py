import unittest
import numpy as np
import src.nn as nn
from src.tensor import Tensor

class Testing(unittest.TestCase):

    # Utility
    def assertEqualUpTo4Decimals(self, expected, actual):
        actual = np.round(actual * np.array(10000)) / 10000
        self.assertEqual(expected, actual.tolist())
    # Utility
    
    # Softmax
    def test_softmax_v(self):
        v = Tensor([-1,0,1])
        r = nn.Softmax()(v)
        self.assertEqualUpTo4Decimals([.09, .2447, .6652], r.data)

    def test_softmax_m(self):
        v = Tensor([[0, 0], [1,0]])
        r = nn.Softmax()(v)
        self.assertEqualUpTo4Decimals([[.5,.5], [.7311, .2689]], r.data)
    
    # Negative log likelihood loss
    def test_neg_log_loss_no_reduction_v(self):
        target = Tensor([0,1,0])
        logits = Tensor([-5, -5, 1])
        probs = nn.Softmax()(logits)

        nll = nn.nlll(probs, target)
        self.assertEqualUpTo4Decimals([6.0049], nll.data)

    def test_neg_log_loss_no_reduction_m(self):
        target = Tensor([[0,1,0], [0,0,1]])
        logits = Tensor([[-5, -5, 1], [-5,-5,1]])
        probs = nn.Softmax()(logits)

        nll = nn.nlll(probs, target)
        self.assertEqualUpTo4Decimals([6.0049, .0049], nll.data)

    def test_neg_log_loss_mean_reduction_m(self):
        target = Tensor([[0,1,0], [0,0,1]])
        logits = Tensor([[-5, -5, 1], [-5,-5,1]])
        probs = nn.Softmax()(logits)

        nll = nn.nlll(probs, target, reduction='mean')
        self.assertEqualUpTo4Decimals([3.0049], nll.data)

    def test_neg_log_loss_sum_reduction_m(self):
        target = Tensor([[0,1,0], [0,0,1]])
        logits = Tensor([[-5, -5, 1], [-5,-5,1]])
        probs = nn.Softmax()(logits)

        nll = nn.nlll(probs, target, reduction='sum')
        self.assertEqualUpTo4Decimals([6.0099], nll.data)