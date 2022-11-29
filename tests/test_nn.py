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

    def test_init_hidden_1(self):
        mlp = nn.MLP(inputs=10, hidden=(20,), outs=5)
        weights = [x.shape for x in mlp.W]
        bias = [x.shape for x in mlp.b]
        self.assertEqual([(10,20), (20,5)], weights)
        self.assertEqual([(20,)], bias)
    
    def test_init_hidden_3(self):
        mlp = nn.MLP(inputs=50, hidden=(40,30), outs=5)
        weights = [x.shape for x in mlp.W]
        bias = [x.shape for x in mlp.b]
        self.assertEqual([(50,40), (40,30), (30,5)], weights)
        self.assertEqual([(40,), (30,)], bias)
    
    def test_softmax_v(self):
        v = Tensor([-1,0,1])
        r = nn.softmax(v)
        self.assertEqualUpTo4Decimals([.09, .2447, .6652], r.data)

    def test_softmax_m(self):
        v = Tensor([[0, 0], [1,0]])
        r = nn.softmax(v)
        self.assertEqualUpTo4Decimals([[.5,.5], [.7311, .2689]], r.data)
    
    def test_neg_log_loss_no_reduction_v(self):
        target = Tensor([0,1,0])
        logits = Tensor([-5, -5, 1])
        probs = nn.softmax(logits)

        nll = nn.nlll(probs, target)
        self.assertEqualUpTo4Decimals([6.0049], nll.data)

    def test_neg_log_loss_no_reduction_m(self):
        target = Tensor([[0,1,0], [0,0,1]])
        logits = Tensor([[-5, -5, 1], [-5,-5,1]])
        probs = nn.softmax(logits)

        nll = nn.nlll(probs, target)
        self.assertEqualUpTo4Decimals([6.0049, .0049], nll.data)
