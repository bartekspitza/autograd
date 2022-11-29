import unittest
import numpy as np
from src.nn import MLP

class Testing(unittest.TestCase):

    def test_hidden_1(self):
        nn = MLP(inputs=10, hidden=(20,), outs=5)
        weights = [x.shape for x in nn.W]
        bias = [x.shape for x in nn.b]
        self.assertEqual([(10,20), (20,5)], weights)
        self.assertEqual([(20,)], bias)
    
    def test_hidden_3(self):
        nn = MLP(inputs=50, hidden=(40,30), outs=5)
        weights = [x.shape for x in nn.W]
        bias = [x.shape for x in nn.b]
        self.assertEqual([(50,40), (40,30), (30,5)], weights)
        self.assertEqual([(40,), (30,)], bias)