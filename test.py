import unittest
import nn
import math

class Testing(unittest.TestCase):

    # Test initialization of Tensors
    # ---------------------------------
    def test_init_s(self):
        s = nn.Tensor(5)
        self.assertEqual((), s.shape)
        self.assertEqual(0, s.dim)
    def test_init_v(self):
        s = nn.Tensor(5); 
        v = nn.Tensor([s, s, s])
        self.assertEqual((3, ), v.shape)
        self.assertEqual(1, v.dim)
    def test_init_m(self):
        s = nn.Tensor(5); 
        v = nn.Tensor([s, s, s])
        m = nn.Tensor([v, v])
        self.assertEqual((2, 3), m.shape)
        self.assertEqual(2, m.dim)
    

    # Test operations
    # ---------------------------------

    # Mult
    def test_mult_ss(self):
        s1 = nn.Tensor(2, requires_grad=True)
        s2 = nn.Tensor(5, requires_grad=True)
        r = s1*s2
        self.assertEqual(10, r.data)

        r.grad=1; r.backward()
        #self.assertEqual([5, 2], [s1.grad.data, s2.grad.data])

    def test_mult_sv(self):
        s = nn.Tensor(2)
        v = nn.wrap([6, 8])
        self.assertEqual([12, 16], (v*s).unwrap())

    def test_mult_sm(self):
        s = nn.wrap(2)
        m = nn.wrap([[12, 8], [16, 32]])
        self.assertEqual([[24, 16], [32, 64]], (m*s).unwrap())

    def test_mult_vv(self):
        a = nn.wrap([1, 2])
        b = nn.wrap([10, -20])
        self.assertEqual([10, -40], (a*b).unwrap())
        self.assertEqual([10, -40], (b*a).unwrap())

    def test_mult_vv_shapes_mismatch_raises(self):
        a = nn.wrap([1])
        b = nn.wrap([10, 20])
        self.assertRaises(RuntimeError, lambda: a*b)

    def test_mult_vm_mv(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([10, 20])
        self.assertEqual([[10, 40], [30, 80]], (a*b).unwrap())
        self.assertEqual([[10, 40], [30, 80]], (b*a).unwrap())

    def test_mult_mv_shapes_mismatch_raises(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a*b)

    def test_mult_mm(self):
        a = nn.wrap([[1, 2], [3, 4]])
        self.assertEqual([[1, 4], [9, 16]], (a*a).unwrap())
    
    # Add
    def test_add_ss(self):
        s1 = nn.Tensor(2, requires_grad=True)
        s2 = nn.Tensor(5, requires_grad=True)
        r = s1+s2
        self.assertEqual(7, r.data)
        r.grad=1; r.backward()
        #self.assertEqual([1, 1], [s1.grad.data, s2.grad.data])

    def test_add_sv(self):
        v = nn.wrap([6, 8])
        s = nn.Tensor(2)
        self.assertEqual([8, 10], (v+s).unwrap())

    def test_add_sm(self):
        m = nn.wrap([[12, 8], [16, 32]])
        s = nn.Tensor(2)
        self.assertEqual([[14, 10], [18, 34]], (m+s).unwrap())

    def test_add_vv(self):
        a = nn.wrap([1, 2])
        b = nn.wrap([10, 20])
        self.assertEqual([11, 22], (a+b).unwrap())

    def test_add_vv_shapes_mismatch_raises(self):
        a = nn.wrap([1])
        b = nn.wrap([10, 20])
        self.assertRaises(RuntimeError, lambda: a+b)

    def test_add_vm(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([10, -20])
        c = b+a
        self.assertEqual([[11, -18], [13, -16]], c.unwrap())

    def test_add_mv(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([10, 20])
        self.assertEqual([[11, 22], [13, 24]], (a+b).unwrap())

    def test_add_mv_shapes_mismatch_raises(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a+b)

    def test_add_mm(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([[5, 6], [7, 8]])
        self.assertEqual([[6,8], [10, 12]], (a+b).unwrap())
        self.assertEqual([[6,8], [10, 12]], (b+a).unwrap())


    # Sub
    def test_sub_ss(self):
        s = nn.Tensor(2)
        self.assertEqual(0, (s-s).data)

    def test_sub_sv_vs(self):
        s = nn.Tensor(2)
        v = nn.wrap([6, 8])
        self.assertEqual([4, 6], (v-s).unwrap())
        self.assertEqual([-4, -6], (s-v).unwrap())

    def test_sub_sm_ms(self):
        s = nn.Tensor(2)
        m = nn.wrap([[12, 8], [16, 32]])
        self.assertEqual([[10, 6], [14, 30]], (m-s).unwrap())
        self.assertEqual([[-10, -6], [-14, -30]], (s-m).unwrap())

    def test_sub_vv(self):
        a = nn.wrap([1, 2])
        b = nn.wrap([10, 20])
        self.assertEqual([-9, -18], (a-b).unwrap())

    def test_sub_vm_mv(self):
        v = nn.wrap([10, 20])
        m = nn.wrap([[1, 2], [3, 4]])
        self.assertEqual([[-9, -18], [-7, -16]], (m-v).unwrap())
        self.assertEqual([[9, 18], [7, 16]], (v-m).unwrap())

    def test_sub_mm(self):
        m1 = nn.wrap([[1,2], [3,4]])
        m2 = nn.wrap([[1, 1], [1,1]])
        self.assertEqual([[0,1], [2,3]], (m1-m2).unwrap())

    def test_sub_vv_shapes_mismatch_raises(self):
        a = nn.wrap([1])
        b = nn.wrap([10, 20])
        self.assertRaises(RuntimeError, lambda: a-b)

    def test_sub_mv_shapes_mismatch_raises(self):
        a = nn.wrap([[1, 2], [3, 4]])
        b = nn.wrap([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a-b)
    
    # Div
    def test_div_s(self):
        s = nn.Tensor(2) / nn.Tensor(5)
        self.assertEqual(2/5, s.unwrap())

    def test_div_sv_vs(self):
        s = nn.Tensor(2)
        v = nn.wrap([8, 10])
        self.assertEqual([4, 5], (v/s).unwrap())
        self.assertEqual([0.25, 0.2], (s/v).unwrap())

    def test_div_sm_ms(self):
        s = nn.Tensor(2)
        m = nn.wrap([[8, 10], [16, 20]])
        self.assertEqual([[4, 5], [8, 10]], (m/s).unwrap())
        self.assertEqual([[0.25, 0.2], [0.125, .10]], (s/m).unwrap())

    def test_div_vv(self):
        a = nn.wrap([12, 8])
        b = nn.wrap([2, 4])
        c = a/b
        self.assertEqual([6, 2], c.unwrap())

    def test_div_vm_mv(self):
        v = nn.wrap([6, 8])
        m = nn.wrap([[12, 8], [18, 32]])
        self.assertEqual([[2, 1], [3, 4]], (m/v).unwrap())
        self.assertEqual([[0.5, 1.0], [1/3, 0.25]], (v/m).unwrap())

    def test_div_mm(self):
        m1 = nn.wrap([[1, 2], [5, 4]])
        m2 = nn.wrap([[5, 4], [4, 16]])
        self.assertEqual([[0.2, 0.5], [1.25, 0.25]], (m1/m2).unwrap())

    def test_div_vv_shapes_mismatch_raises(self):
        a = nn.wrap([1])
        b = nn.wrap([10, 20])
        self.assertRaises(RuntimeError, lambda: a/b)
    
    # Dot
    def test_dot_vv(self):
        a = nn.wrap([1, 2])
        b = nn.wrap([10, 20])
        self.assertEqual(50, (a@b).unwrap())
        self.assertEqual(50, (b@a).unwrap())
    
    def test_dot_vv_shapes_mismatch_raises(self):
        a = nn.wrap([1])
        b = nn.wrap([10, 20])
        self.assertRaises(RuntimeError, lambda: a-b)
    
    def test_dot_mv(self):
        m = nn.wrap([[1, 2], [3, 4]])
        v = nn.wrap([10, 20])
        c = m@v
        self.assertEqual([50, 110], c.unwrap())

    def test_dot_mv_shapes_mismatch_raises(self):
        m = nn.wrap([[1, 2], [3, 4]])
        v = nn.wrap([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: m@v)
    
    def test_dot_vm(self):
        # 3 * 3x2
        v = nn.wrap([7, 2, 3])
        m = nn.wrap([[1, 2], [3, 4], [5, 6]])
        c = v@m
        self.assertEqual([28, 40], c.unwrap())
        # 3 * 3x3
        v = nn.wrap([1, 2])
        m = nn.wrap([[10, 20, 5], [30, 40, 20]])
        c = v@m
        self.assertEqual([70, 100, 45], c.unwrap())
        # 3 * 3x4
        v = nn.wrap([1, 2])
        m = nn.wrap([[10, 20, 5, 30], [30, 40, 20, 15]])
        c = v@m
        self.assertEqual([70, 100, 45, 60], c.unwrap())
    
    def test_dot_vm_shapes_mismatch_raises(self):
        v = nn.wrap([10, 20])
        m = nn.wrap([[1, 2], [3, 4], [5,6]])
        self.assertRaises(RuntimeError, lambda: v@m)
    
    def test_dot_mm(self):
        x = nn.wrap([
            [7, 2, 3], 
            [6, 2, 7]
            ])
        m = nn.wrap([[1, 2], [3, 4], [5,6]])
        c = x@m
        self.assertEqual([28, 40], c[0].unwrap())
        self.assertEqual([47, 62], c[1].unwrap())
    
    # exp(x)
    def test_exp_s(self):
        s = nn.Tensor(0)
        self.assertAlmostEqual(1.0, s.exp().unwrap())

    def test_exp_v(self):
        v = nn.wrap([-1, 0, 1, 2])
        res = v.exp()
        self.assertAlmostEqual(0.3678, res.unwrap()[0], delta=0.0001)
        self.assertAlmostEqual(1.0, res.unwrap()[1])
        self.assertAlmostEqual(math.e, res.unwrap()[2])
    
    def test_exp_m(self):
        m = nn.wrap([[0, 1], [1, 0]])
        res = m.exp().unwrap()
        self.assertEqual([[1, math.e], [math.e, 1]], res)


    # log(x)
    def test_log_s(self):
        s = nn.Tensor(1)
        self.assertEqual(0, s.log().unwrap())

    def test_log_v(self):
        v = nn.wrap([-1, 0, 1, 0.3678])
        res = v.log()
        self.assertEqual([math.nan, math.nan, 0], res.unwrap()[:3])
        self.assertAlmostEqual(-1, res.unwrap()[3], delta=0.01)

    def test_log_m(self):
        m = nn.wrap([[1, math.e], [math.e, 1]])
        res = m.log().unwrap()
        self.assertEqual([[0, 1], [1, 0]], res)
    
    # tanh(x)
    def test_tanh_s(self):
        s = nn.Tensor(0)
        self.assertAlmostEqual(0, nn.tanh(s).unwrap())
    
    def test_tanh_v(self):
        v = nn.wrap([-1, 0, 1])
        res = nn.tanh(v).unwrap()
        self.assertAlmostEqual(-0.7616, res[0], delta=0.00001)
        self.assertAlmostEqual(0, res[1])
        self.assertAlmostEqual(0.7616, res[2], delta=0.00001)
    
    def test_tanh_m(self):
        v = nn.wrap([[-1, 0, 1]])
        res = nn.tanh(v)
        res = res.unwrap()[0]
        self.assertAlmostEqual(-0.7616, res[0], delta=0.00001)
        self.assertAlmostEqual(0, res[1])
        self.assertAlmostEqual(0.7616, res[2], delta=0.00001)
    """

    # Backprop
    # ---------------------------------
    # Add
    def test_grad_add_vec_and_vec(self):
        a = nn.wrap([2, -2], requires_grad=True)
        b = nn.wrap([1, -1], requires_grad=True)
        c = a+b; c.grad=1
        c.backward()
        self.assertEqual([1, 1], a.grad.unwrap())
        self.assertEqual([1, 1], b.grad.unwrap())

    def test_grad_add_mat_and_vec(self):
        a = nn.wrap([[1,1], [2,2]], requires_grad=True)
        b = nn.wrap([2, 2], requires_grad=True)
        c = a+b; c.grad=1
        c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.unwrap())
        self.assertEqual([2,2], b.grad.unwrap())
    
    def test_grad_add_vec_and_mat(self):
        a = nn.wrap([[1,1], [2,2]], requires_grad=True)
        b = nn.wrap([2, 2], requires_grad=True)
        c = b+a; c.grad=1; c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.unwrap())
        self.assertEqual([2,2], b.grad.unwrap())

    def test_grad_add_mat_and_mat(self):
        a = nn.wrap([[1, 2], [3, 4]], requires_grad=True)
        b = nn.wrap([[5, 6], [7, 8]], requires_grad=True)
        c = a+b; c.grad=1; c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.unwrap())
        self.assertEqual([[1,1], [1,1]], b.grad.unwrap())

        a.grad=0; b.grad=0
        c = b+a; c.grad=1; c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.unwrap())
        self.assertEqual([[1,1], [1,1]], b.grad.unwrap())
    
    # Mult
    def test_grad_mult_vec_and_vec(self):
        a = nn.wrap([2, 2], requires_grad=True)
        b = nn.wrap([4, 4], requires_grad=True)
        c = a*b; c.grad=1
        c.backward()
        self.assertEqual([4, 4], a.grad.unwrap())
        self.assertEqual([2, 2], b.grad.unwrap())
    
    def test_grad_mult_mat_and_vec(self):
        a = nn.wrap([[1,2], [3, 4]], requires_grad=True)
        b = nn.wrap([5,6], requires_grad=True)
        c = a*b; c.grad=1; c.backward()

        self.assertEqual([[5, 6], [5, 6]], a.grad.unwrap())
        self.assertEqual([4, 6], b.grad.unwrap())
    
    def test_grad_mult_vec_and_mat(self):
        a = nn.wrap([[1,2], [3, 4]], requires_grad=True)
        b = nn.wrap([5,6], requires_grad=True)
        c = b*a; c.grad=1; c.backward()

        self.assertEqual([[5, 6], [5, 6]], a.grad.unwrap())
        self.assertEqual([4, 6], b.grad.unwrap())
    
    def test_grad_mult_mat_and_mat(self):
        a = nn.wrap([[1,2], [3, 4]], requires_grad=True)
        b = nn.wrap([[5,6], [7,8]], requires_grad=True)
        c = b*a; c.grad=1; c.backward()

        self.assertEqual([[5, 6], [7, 8]], a.grad.unwrap())
        self.assertEqual([[1, 2], [3, 4]], b.grad.unwrap())
    
    def test_grad_mult_vec_scalar(self):
        a = nn.wrap([1,2, 3], requires_grad=True)
        #b = nn.Tensor(5, requires_grad=True)
        #print('asdfasdfasdf')
        #c = a*b; 
        #c.grad=1; 
        #c.backward()

        #self.assertEqual([5, 5, 5], a.grad.unwrap())
        #self.assertEqual([6], b.grad.unwrap())
    
    # Div
    def test_grad_div_vec_and_vec(self):
        x1 = nn.wrap([2, 5, 5,   0.5], requires_grad=True)
        x2 = nn.wrap([5, 2, 0.5, 5  ], requires_grad=True)
        c = x1/x2; c.grad=1; c.backward()

        self.assertEqual([0.2, 0.5, 2.0, 0.2], x1.grad.unwrap())
        self.assertEqual([-0.08, -1.25, -20, -0.02], x2.grad.unwrap())

    def test_grad_div_mat_and_vec(self):
        m = nn.wrap([[5], [4]], requires_grad=True)
        v = nn.wrap([2], requires_grad=True)
        c = m/v; c.grad=1; c.backward()

        self.assertEqual([[0.5], [0.5]], m.grad.unwrap())
        self.assertEqual([-2.25], v.grad.unwrap()) # also tests so that gradients are accumulated
    
    def test_grad_div_vec_and_mat(self):
        m = nn.wrap([[5], [4]], requires_grad=True)
        v = nn.wrap([2], requires_grad=True)
        c = v/m; c.grad=1; c.backward()

        self.assertEqual([[-0.08], [-0.125]], m.grad.unwrap())
        self.assertEqual([0.45], v.grad.unwrap()) 
    
    def test_grad_div_mat_and_mat(self):
        m1 = nn.wrap([[2, 5], [5,   0.5]], requires_grad=True)
        m2 = nn.wrap([[5, 2,], [0.5, 5]], requires_grad=True)
        c = m1/m2; c.grad=1; c.backward()

        self.assertEqual([[0.2, 0.5], [2.0, 0.2]], m1.grad.unwrap())
        self.assertEqual([[-0.08, -1.25], [-20, -0.02]], m2.grad.unwrap())
    
    ## Sub
    def test_grad_sub_vec_and_vec(self):
        x1 = nn.wrap([-1, 0, 1], requires_grad=True)
        x2 = nn.wrap([-1, 0, 1], requires_grad=True)
        c = x1-x2; c.grad=1; c.backward()

        self.assertEqual([1, 1, 1], x1.grad.unwrap())
        self.assertEqual([-1, -1, -1], x2.grad.unwrap())
    
    def test_grad_sub_mat_and_vec(self):
        m = nn.wrap([[5], [5]], requires_grad=True)
        v = nn.wrap([1], requires_grad=True)
        c = m-v; c.grad=1; c.backward()
        self.assertEqual([[1], [1]], m.grad.unwrap())
        self.assertEqual([-2], v.grad.unwrap())
    
    def test_grad_sub_vec_and_mat(self):
        m = nn.wrap([[5, 5], [5, 5]], requires_grad=True)
        v = nn.wrap([1, 1], requires_grad=True)
        c = v-m; c.grad=1; c.backward()
        self.assertEqual([[-1, -1], [-1, -1]], m.grad.unwrap())
        self.assertEqual([2, 2], v.grad.unwrap())

    def test_grad_sub_mat_and_mat(self):
        m = nn.wrap([[5], [5]], requires_grad=True)
        m1 = nn.wrap([[5], [5]], requires_grad=True)
        c = m-m1; c.grad=1; c.backward()
        self.assertEqual([[1], [1]], m.grad.unwrap())
        self.assertEqual([[-1], [-1]], m1.grad.unwrap())
    
    ## Dot
    def test_grad_dot_vec_and_vec(self):
        x1 = nn.wrap([2, 3, 4], requires_grad=True)
        x2 = nn.wrap([5, 6, 7], requires_grad=True)
        #c = x1@x2; c.grad=1; c.backward()

        #self.assertEqual([1, 1, 1], x1.grad.unwrap())
        #self.assertEqual([-1, -1, -1], x2.grad.unwrap())
    """


    # Other stuff, subscript, getitem, setitem etc
    # ---------------------------------
    def test_getitem_vector(self):
        m = nn.wrap([1,2])
        self.assertEqual(1, m[0].unwrap())
        self.assertEqual(2, m[1].unwrap())

    def test_getitem_matrix(self):
        m = nn.wrap([[1,2], [3,4]])
        self.assertEqual([1,2], m[0].unwrap())
        self.assertEqual([3,4], m[1].unwrap())

    def test_getmultipleitem_vector(self):
        m = nn.wrap([1,2])
        self.assertEqual([2, 1], m[[1, 0]].unwrap())

    def test_getmultipleitem_matrix(self):
        m = nn.wrap([[1,2], [3,4]])
        self.assertEqual([[3,4], [1,2]], m[[1, 0]].unwrap())

    def test_setitem_vector(self):
        v = nn.wrap([1, 2])
        v[0] = 5
        self.assertEqual(5, v[0])

    def test_setitem_matrix(self):
        m = nn.wrap([[1,2], [3,4]])
        m[0] = nn.wrap([10, 20])
        self.assertEqual([10, 20], m[0].unwrap())

    # Other functions
    # ---------------------------------
    def test_sum_v(self):
        t = nn.wrap([1, 2, 3])
        self.assertEqual(6, nn.sum(t).unwrap())

    def test_sum_m(self):
        t = nn.wrap([[1,2], [3,4]])
        c = nn.sum(t)
        self.assertEqual(10, c.unwrap())
    
    def test_ones(self):
        a = nn.ones((3,)).unwrap()

        self.assertEqual([1, 1, 1], a)
        a = nn.ones((3, 5)).unwrap()
        self.assertEqual(3, len(a))
        self.assertEqual(5, len(a[0]))
    
    ## unwrap
    def test_unwrap_s(self):
        s = nn.Tensor(5)
        r = s.unwrap()
        self.assertEqual(type(r), int)
        self.assertEqual(5, r)
    def test_unwrap_v(self):
        v = nn.Tensor([nn.Tensor(2), nn.Tensor(5)])
        r = v.unwrap()
        self.assertEqual(type(r), list)
        self.assertEqual(type(r[0]), int)
    def test_unwrap_m(self):
        v = nn.Tensor([nn.Tensor(2), nn.Tensor(5)])
        m = nn.Tensor([v, v, v]) 
        r = m.unwrap()
        self.assertEqual(type(r), list)
        self.assertEqual(type(r[0]), list)
        self.assertEqual(type(r[0][0]), int)
    
    ## wrap
    def test_wrap_s(self):
        s = 5
        tensor= nn.wrap(s)
        self.assertEqual(type(tensor), nn.Tensor)
        self.assertEqual(tensor.shape, ())
    def test_wrap_v(self):
        v = [1, 2, 3]
        tensor= nn.wrap(v)
        self.assertEqual(type(tensor), nn.Tensor)
        self.assertEqual(type(tensor[0]), nn.Tensor)
        self.assertEqual(tensor.shape, (3,))
    def test_wrap_m(self):
        v = [1, 2, 3]
        m = [v, v]
        tensor= nn.wrap(m)
        self.assertEqual(type(tensor), nn.Tensor)
        self.assertEqual(type(tensor[0]), nn.Tensor)
        self.assertEqual(type(tensor[0][0]), nn.Tensor)
        self.assertEqual(tensor.shape, (2,3))
    def test_wrap_m3dim(self):
        v = [1, 2, 3]
        m = [v, v]
        m3 = [m, m, m, m]
        tensor= nn.wrap(m3)
        self.assertEqual(type(tensor), nn.Tensor)
        self.assertEqual(type(tensor[0]), nn.Tensor)
        self.assertEqual(type(tensor[0][0]), nn.Tensor)
        self.assertEqual(type(tensor[0][0][0]), nn.Tensor)
        self.assertEqual(tensor.shape, (4, 2, 3))
    def test_wrap_m_throwsWhenDifferentSizesInSameDim(self):
        v = [1, 2, 3]
        m = [v, (v+[1])]
        self.assertRaises(RuntimeError, lambda: nn.wrap(m))
    def test_wrap_ofNestedTensor(self):
        v = [1, 2, 3]
        m = [v, v]
        m3 = [m, m, m, m]
        tensor= nn.wrap(m3)
        self.assertEqual((2,3), tensor[0].shape)


if __name__ == '__main__':
    unittest.main()