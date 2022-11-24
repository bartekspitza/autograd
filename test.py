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

    def test_getitem(self):
        a = nn.Tensor([3,4])
        self.assertEqual(3, a[0].data)


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
    def test_grad_vm(self):
        m = nn.Tensor([[6,4,2], [1,2,3]], requires_grad=True)
        v = nn.Tensor([1, 2], requires_grad=True)
        r = v@m; r.backward()

        self.assertEqual([12, 6], v.grad.tolist())
        self.assertEqual([[1,1,1], [2,2,2]], m.grad.tolist())

    
if __name__ == '__main__':
    unittest.main()