import unittest
import numpy as np
from src import tensor as nn

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
    
    # Mult
    def test_grad_mult_ss(self):
        s1 = nn.Tensor(5, requires_grad=True)
        s2 = nn.Tensor(3, requires_grad=True)
        r = s1*s2
        r.grad=4 
        r.backward()

        self.assertEqual(12, s1.grad.item())
        self.assertEqual(20, s2.grad.item())

    def test_grad_mult_sv(self):
        s = nn.Tensor(5, requires_grad=True)
        v = nn.Tensor([3,4], requires_grad=True)
        r = s*v
        r.grad=4 
        r.backward()

        self.assertEqual(12+16, s.grad.item())
        self.assertEqual([20,20], v.grad.tolist())

    def test_grad_mult_vs(self):
        s = nn.Tensor(5, requires_grad=True)
        v = nn.Tensor([3,4], requires_grad=True)
        r = v*s
        r.grad=4 
        r.backward()

        self.assertEqual(28, s.grad.item())
        self.assertEqual([20,20], v.grad.tolist())
    
    def test_grad_mult_sm(self):
        s = nn.Tensor(5, requires_grad=True)
        m = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = s*m
        r.grad=np.array([[3,4], [1, 2]]) 
        r.backward()

        self.assertEqual(22, s.grad.item())
        self.assertEqual([[15, 20], [5, 10]], m.grad.tolist())
    def test_grad_mult_ms(self):
        s = nn.Tensor(5, requires_grad=True)
        m = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = m*s
        r.grad=np.array([[3,4], [1, 2]]) 
        r.backward()

        self.assertEqual(22, s.grad.item())
        self.assertEqual([[15, 20], [5, 10]], m.grad.tolist())

    def test_grad_mult_vv(self):
        v1 = nn.Tensor([1,2], requires_grad=True)
        v2 = nn.Tensor([3,4], requires_grad=True)
        r = v1*v2
        r.grad=np.array([3,4], dtype=float); 
        r.backward()

        self.assertEqual([9, 16], v1.grad.tolist())
        self.assertEqual([3, 8], v2.grad.tolist())
    
    def test_grad_mult_vm(self):
        v = nn.Tensor([1,2], requires_grad=True)
        m = nn.Tensor([[2,1], [3,4]], requires_grad=True)
        r = v*m
        r.grad=np.array([[4,3], [2,1]], dtype=float); 
        r.backward()

        self.assertEqual([14, 7], v.grad.tolist())
        self.assertEqual([[4, 6], [2, 2]], m.grad.tolist())

    def test_grad_mult_mv(self):
        v = nn.Tensor([1,2], requires_grad=True)
        m = nn.Tensor([[2,1], [3,4]], requires_grad=True)
        r = m*v
        r.grad=np.array([[4,3], [2,1]], dtype=float); 
        r.backward()

        self.assertEqual([14, 7], v.grad.tolist())
        self.assertEqual([[4, 6], [2, 2]], m.grad.tolist())

    def test_grad_mult_mm(self):
        m1 = nn.Tensor([[1,2],[3,4]], requires_grad=True)
        m2 = nn.Tensor([[5,6],[7,8]], requires_grad=True)
        r = m1*m2
        r.grad=np.array([[3,4],[1,2]], dtype=float); 
        r.backward()

        self.assertEqual([[15, 24], [7, 16]], m1.grad.tolist())
        self.assertEqual([[3, 8], [3, 8]], m2.grad.tolist())
    
    # sub
    def test_grad_sub_ss(self):
        s1 = nn.Tensor(5, requires_grad=True)
        s2 = nn.Tensor(3, requires_grad=True)
        r = s1-s2; r.grad=5; r.backward()

        self.assertEqual(5, s1.grad.item())
        self.assertEqual(-5, s2.grad.item())

    def test_grad_sub_sv(self):
        s = nn.Tensor(5, requires_grad=True)
        v = nn.Tensor([1,2], requires_grad=True)
        r = s-v 
        r.grad=np.array([3,4])
        r.backward()

        self.assertEqual(7, s.grad.item())
        self.assertEqual([-3,-4], v.grad.tolist())

    def test_grad_sub_vs(self):
        s = nn.Tensor(5, requires_grad=True)
        v = nn.Tensor([1,2], requires_grad=True)
        r = v-s
        r.grad=np.array([3,4])
        r.backward()

        self.assertEqual(-7, s.grad.item())
        self.assertEqual([3,4], v.grad.tolist())

    def test_grad_sub_sm(self):
        s = nn.Tensor(5, requires_grad=True)
        m = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = s-m
        r.grad=np.array([[1,2], [3,4]])
        r.backward()

        self.assertEqual(10, s.grad.item())
        self.assertEqual([[-1, -2], [-3, -4]], m.grad.tolist())

    def test_grad_sub_ms(self):
        s = nn.Tensor(5, requires_grad=True)
        m = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = m-s
        r.grad=np.array([[1,2], [3,4]])
        r.backward()

        self.assertEqual(-10, s.grad.item())
        self.assertEqual([[1, 2], [3, 4]], m.grad.tolist())

    def test_grad_sub_vv(self):
        v1 = nn.Tensor([1,3,5], requires_grad=True)
        v2 = nn.Tensor([1,3,5], requires_grad=True)
        r=v1-v2
        r.grad = np.array([3,2,1])
        r.backward()

        self.assertEqual([3,2,1], v1.grad.tolist())
        self.assertEqual([-3,-2,-1], v2.grad.tolist())
    
    def test_grad_sub_vm(self):
        v = nn.Tensor([1,3], requires_grad=True)
        m = nn.Tensor([[1,2], [1,2]], requires_grad=True)
        r=v-m
        r.grad = np.array([[1,2],[3,4]])
        r.backward()

        self.assertEqual([4,6], v.grad.tolist())
        self.assertEqual([[-1,-2],[-3,-4]], m.grad.tolist())

    def test_grad_sub_mv(self):
        v = nn.Tensor([1,3], requires_grad=True)
        m = nn.Tensor([[1,2], [1,2]], requires_grad=True)
        r=m-v
        r.grad = np.array([[1,2],[3,4]])
        r.backward()

        self.assertEqual([-4,-6], v.grad.tolist())
        self.assertEqual([[1,2],[3,4]], m.grad.tolist())

    def test_grad_sub_mm(self):
        m1 = nn.Tensor([[1,2], [1,2]], requires_grad=True)
        m2 = nn.Tensor([[1,2], [1,2]], requires_grad=True)
        r=m1-m2
        r.grad = np.array([[1,2], [3,4]])
        r.backward()

        self.assertEqual([[1,2], [3,4]], m1.grad.tolist())
        self.assertEqual([[-1,-2], [-3,-4]], m2.grad.tolist())
    
    # Div
    def test_grad_div_ss(self):
        s1 = nn.Tensor(4, requires_grad=True)
        s2 = nn.Tensor(2, requires_grad=True)
        r=s1/s2;r.grad=3;r.backward()

        self.assertEqual(1.5, s1.grad.item())
        self.assertEqual(-3, s2.grad.item())

    def test_grad_div_sv(self):
        s = nn.Tensor(4, requires_grad=True)
        v = nn.Tensor([4,4], requires_grad=True)
        r=s/v;r.grad=np.array([1,2]);r.backward()

        self.assertEqual(0.75, s.grad.item())
        self.assertEqual([-.25, -.5], v.grad.tolist())

    def test_grad_div_vs(self):
        s = nn.Tensor(4, requires_grad=True)
        v = nn.Tensor([4,4], requires_grad=True)
        r=v/s;r.grad=np.array([1,2]);r.backward()

        self.assertEqual(-0.75, s.grad.item())
        self.assertEqual([.25, 0.5], v.grad.tolist())
    
    def test_grad_div_sm(self):
        s = nn.Tensor(4, requires_grad=True)
        m = nn.Tensor([[2,4], [8,16]], requires_grad=True)
        r=s/m
        r.grad=np.array([[1,2],[3,4]])
        r.backward()

        self.assertEqual(1.625, s.grad.item())
        self.assertEqual([[-1,-.5], [-.1875, -.0625]], m.grad.tolist())

    def test_grad_div_ms(self):
        s = nn.Tensor(4, requires_grad=True)
        m = nn.Tensor([[2,4], [8,16]], requires_grad=True)
        r=m/s
        r.grad=np.array([[1,2],[3,4]])
        r.backward()

        self.assertEqual(-6.125, s.grad.item())
        self.assertEqual([[.25, .5], [.75, 1]], m.grad.tolist())
    
    def test_grad_div_vv(self):
        v1 = nn.Tensor([1,2], requires_grad=True)
        v2 = nn.Tensor([4,4], requires_grad=True)
        r=v1/v2
        r.grad=np.array([1,2])
        r.backward()

        self.assertEqual([0.25, 0.5], v1.grad.tolist())
        self.assertEqual([-0.0625, -0.25], v2.grad.tolist())
    
    def test_grad_div_vm(self):
        v = nn.Tensor([1,2], requires_grad=True)
        m = nn.Tensor([[2,4], [8,16]], requires_grad=True)
        r=v/m
        r.grad=np.array([[2,1],[4,8]])
        r.backward()

        self.assertEqual([1.5, .75], v.grad.tolist())
        self.assertEqual([[-.5, -.125], [-.0625, -.0625]], m.grad.tolist())

    def test_grad_div_mv(self):
        v = nn.Tensor([1,2], requires_grad=True)
        m = nn.Tensor([[2,4], [8,16]], requires_grad=True)
        r=m/v
        r.grad=np.array([[2,1],[4,8]])
        r.backward()

        self.assertEqual([-36, -33], v.grad.tolist())
        self.assertEqual([[2, .5], [4, 4]], m.grad.tolist())

    def test_grad_div_mm(self):
        m1 = nn.Tensor([[1,2], [2,1]], requires_grad=True)
        m2 = nn.Tensor([[2,4], [8,10]], requires_grad=True)
        r=m1/m2
        r.grad=np.array([[1,2], [3,4]])
        r.backward()

        self.assertEqual([[.5, .5], [.375, .4]], m1.grad.tolist())
        self.assertEqual([[-.25, -.25], [-.09375, -.04]], m2.grad.tolist())
    
    def test_grad_div_mm(self):
        m1 = nn.Tensor([[4, 4], [8,8]], requires_grad=True)
        m2 = nn.Tensor([[4], [8]], requires_grad=True)
        r=m1/m2
        r.grad=np.array([[1,2], [3,4]])
        r.backward()

        self.assertEqual([[.25, .5], [0.375, .5]], m1.grad.tolist())
        self.assertEqual([[-.75], [-.875]], m2.grad.tolist())

    # Matmul
    def test_grad_matmul_vv(self):
        v1 = nn.Tensor([1,2], requires_grad=True)
        v2 = nn.Tensor([3,4], requires_grad=True)
        r = v1@v2; r.grad=2; r.backward()

        self.assertEqual([6, 8], v1.grad.tolist())
        self.assertEqual([2, 4], v2.grad.tolist())

    def test_grad_matmul_mv(self):
        v = nn.Tensor([1,2], requires_grad=True)
        m = nn.Tensor([[2,1], [3, 4]], requires_grad=True)

        r = m@v
        r.grad=np.array([3, 4])
        r.backward()

        self.assertEqual([18, 19], v.grad.tolist())
        self.assertEqual([[3,6], [4, 8]], m.grad.tolist())

    def test_grad_matmul_vm(self):
        v = nn.Tensor([1,2], requires_grad=True)
        m = nn.Tensor([[2,1], [3, 4]], requires_grad=True)

        r = v@m
        r.grad=np.array([3, 4], dtype=float)
        r.backward()

        self.assertEqual([10, 25], v.grad.tolist())
        self.assertEqual([[3,4], [6,8]], m.grad.tolist())
        
    def test_grad_matmul_mm(self):
        m1 = nn.Tensor([[1,2,3], [3,4,5]], requires_grad=True)
        m2 = nn.Tensor([[1,2,3,4], [3,4,5,6], [6,7,8,9]], requires_grad=True)
        r = m1@m2

        r.grad = np.array([[1,2,3,4],[5,6,7,8]])
        r.backward()

        self.assertEqual([[30,50,80],[70,122,200]], m1.grad.tolist())
        self.assertEqual([[16,20,24,28], [22,28,34,40], [28,36,44,52]], m2.grad.tolist())

        """
        Keeping this simpler case around if I ever need to debug this..
        m1 = nn.Tensor([[1,2]], requires_grad=True)
        m2 = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = m1@m2

        r.grad = np.array([[1,2]])
        r.backward()

        self.assertEqual([[5, 11]], m1.grad.tolist())
        self.assertEqual([[1,2],[2,4]], m2.grad.tolist())
        """
    
    # Tanh
    def test_grad_tanh_s(self):
        s = nn.Tensor(0.6, requires_grad=True)
        r = s.tanh(); r.grad = 2; 
        r.backward()
        self.assertEqualUpTo4Decimals([1.4232], s.grad)

    def test_grad_tanh_v(self):
        v = nn.Tensor([0.6, 0.6], requires_grad=True)
        r = v.tanh(); r.grad = np.array([1, 2]); r.backward()
        self.assertEqualUpTo4Decimals([.7116, 1.4232], v.grad)

    def test_grad_tanh_m(self):
        m = nn.Tensor([[.6, .6], [.6, .6]], requires_grad=True)
        r = m.tanh()
        r.grad = np.array([[1,2], [3,4]])
        r.backward()
        self.assertEqualUpTo4Decimals([[.7116, 1.4232], [2.1347, 2.8463]], m.grad)
    
    # exp
    def test_grad_exp_m(self):
        m = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = m.exp()
        r.grad = np.array([[1,2], [3,4]])
        r.backward()
        self.assertEqualUpTo4Decimals([[2.7183,14.7781],[60.2566,218.3926]], m.grad)

    # log
    def test_grad_log_m(self):
        m = nn.Tensor([[1,2], [3,4]], requires_grad=True)
        r = m.log()
        r.grad = np.array([[2,2], [2,2]])
        r.backward()
        self.assertEqualUpTo4Decimals([[2., 1.],[0.6667, .5]], m.grad)

    def assertEqualUpTo4Decimals(self, expected, actual):
        actual = np.round(actual * np.array(10000)) / 10000
        self.assertEqual(expected, actual.tolist())
    
    # sum
    def test_grad_sum_v(self):
        v = nn.Tensor([0.6, 0.6], requires_grad=True)
        r = v.sum(); r.grad=np.array(2); r.backward()
        self.assertEqual([2,2], v.grad.tolist())

    def test_grad_sum_m(self):
        m = nn.Tensor([[1,1], [1,1]], requires_grad=True)
        r = m.sum(); r.grad=np.array(2); r.backward()
        self.assertEqual([[2,2],[2,2]], m.grad.tolist())

    def test_grad_sum_m_axis0(self):
        m = nn.Tensor([[1,1], [1,1]], requires_grad=True)
        r = m.sum(axis=0); r.grad=np.array([3,4]); r.backward()
        self.assertEqual([[3,4],[3,4]], m.grad.tolist())

    def test_grad_sum_m_axis1(self):
        m = nn.Tensor([[1,1], [1,1]], requires_grad=True)
        r = m.sum(axis=1); r.grad=np.array([3,4]); r.backward()
        self.assertEqual([[3,3],[4,4]], m.grad.tolist())
    
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