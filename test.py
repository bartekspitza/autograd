import unittest
import nn
import math

class Testing(unittest.TestCase):

    def test_init(self):
        nn.Tensor(3)
        nn.Tensor([3,3])
        nn.Tensor([[3,3]])
    
    def test_get_dim(self):
        a = nn.Tensor([3,3])
        self.assertEqual(1, a.dim)

    def test_get_shape(self):
        a = nn.Tensor([3,3])
        self.assertEqual((2,), a.shape)


    ## Ops
    def test_add(self):
        a = nn.Tensor([3,3])
        self.assertEqual([6,6], (a+a).tolist())
    def test_mult(self):
        a = nn.Tensor([3,3])
        self.assertEqual([9,9], (a*a).tolist())
    def test_sub(self):
        a = nn.Tensor([3,3])
        self.assertEqual([0,0], (a-a).tolist())
    def test_div(self):
        a = nn.Tensor([3,3])
        self.assertEqual([1,1], (a/a).tolist())
    def test_matmul(self):
        a = nn.Tensor([3,3])
        self.assertEqual(18, (a@a).tolist())

    
if __name__ == '__main__':
    unittest.main()