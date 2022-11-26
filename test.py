import unittest
import nn
import numpy as np
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

    def test_getitem(self):
        a = nn.Tensor([3,4])
        self.assertEqual(3, a[0].data)
    def test_set_grad_s(self):
        s = nn.Tensor(3)
        s.grad = 1
        self.assertEqual([1], s.grad.tolist())
    def test_set_grad_v(self):
        v = nn.Tensor([1,3])
        v.grad = 1
        self.assertEqual([1,1], v.grad.tolist())
    def test_set_grad_m(self):
        v = nn.Tensor([[1,3], [5,6]])
        v.grad = 1
        self.assertEqual([[1,1],[1,1]], v.grad.tolist())


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
        self.assertEqual([18], (a@a).tolist())
    
    ## Backprop
    def test_add_ss(self):
        s = nn.Tensor(3, requires_grad=True)
        r = s+s
        r.grad=1;r.backward()
        self.assertEqual(1, s.grad.data)

    def test_grad_vm(self):
        m = nn.Tensor([[6,4,2], [1,2,3]], requires_grad=True)
        v = nn.Tensor([1, 2], requires_grad=True)
        r = v@m; r.backward()

        self.assertEqual([12, 6], v.grad.tolist())
        self.assertEqual([[1,1,1], [2,2,2]], m.grad.tolist())

    
if __name__ == '__main__':
    unittest.main()