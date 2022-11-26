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
    # Add
    def test_grad_add_ss(self):
        s = nn.Tensor(3, requires_grad=True)
        r = s+s; r.grad=1; r.backward()
        self.assertEqual(2.0, s.grad.item())

    def test_grad_add_sv(self):
        a = nn.Tensor(2, requires_grad=True)
        b = nn.Tensor([1,3,5], requires_grad=True)
        r = a+b; r.grad=1; r.backward()

        self.assertEqual(3.0, a.grad.item())
        self.assertEqual([1,1,1], b.grad.tolist())

    def test_grad_add_vs(self):
        a = nn.Tensor(2, requires_grad=True)
        b = nn.Tensor([1,3,5], requires_grad=True)
        r = b+a; r.grad=1;r.backward()

        self.assertEqual(3.0, a.grad.item())
        self.assertEqual([1,1,1], b.grad.tolist())

    def test_grad_add_sm(self):
        a = nn.Tensor(2, requires_grad=True)
        b = nn.Tensor([[1,1],[1,1]], requires_grad=True)
        r = a+b; r.grad=1; r.backward()

        self.assertEqual(4.0, a.grad.item())
        self.assertEqual([[1,1], [1,1]], b.grad.tolist())
    
    def test_grad_add_ms(self):
        a = nn.Tensor(2, requires_grad=True)
        b = nn.Tensor([[1,1],[1,1]], requires_grad=True)
        r = b+a; r.grad=1; r.backward()

        self.assertEqual(4.0, a.grad.item())
        self.assertEqual([[1,1], [1,1]], b.grad.tolist())

    def test_grad_add_vv(self):
        a = nn.Tensor([1,3,5], requires_grad=True)
        r = a+a; r.grad=2; r.backward()
        self.assertEqual([4,4,4], a.grad.tolist())

    def test_grad_add_vm(self):
        a = nn.Tensor([1,2], requires_grad=True)
        b = nn.Tensor([[1,1], [1,1]], requires_grad=True)
        r = a+b; 
        r.grad=np.array([[1,2], [1,2]], dtype=float); 
        r.backward()

        self.assertEqual([2, 4], a.grad.tolist())
        self.assertEqual([[1,2], [1,2]], b.grad.tolist())

    def test_grad_add_mv(self):
        a = nn.Tensor([1,2], requires_grad=True)
        b = nn.Tensor([[1,1], [1,1]], requires_grad=True)
        r = b+a; 
        r.grad=np.array([[1,2], [1,2]], dtype=float); 
        r.backward()

        self.assertEqual([2, 4], a.grad.tolist())
        self.assertEqual([[1,2], [1,2]], b.grad.tolist())

    def test_grad_add_mm(self):
        b = nn.Tensor([[1,1], [1,1]], requires_grad=True)
        r = b+b
        r.grad=np.array([[1,2], [3,4]], dtype=float); 
        r.backward()

        self.assertEqual([[2,4], [6,8]], b.grad.tolist())

    # matmul
    def test_grad_matmul_vm(self):
        m = nn.Tensor([[6,4,2], [1,2,3]], requires_grad=True)
        v = nn.Tensor([1, 2], requires_grad=True)
        r = v@m; r.backward()

        self.assertEqual([12, 6], v.grad.tolist())
        self.assertEqual([[1,1,1], [2,2,2]], m.grad.tolist())
    
    ## Backprop chaining
    """
    def test_add_sv(self):
        a = nn.Tensor(2, requires_grad=True)
        b = nn.Tensor([1,3,5], requires_grad=True)
        c = nn.Tensor([2,5,4], requires_grad=True)
        r = (a+b)*c; r.grad=1; r.backward()

        self.assertEqual(11.0, a.grad.item())
        self.assertEqual([2,5,4], b.grad.item())
        self.assertEqual([3,5,6], c.grad.item())
    """

    
if __name__ == '__main__':
    unittest.main()