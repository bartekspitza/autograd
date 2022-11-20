import unittest
import nn
import math

class Testing(unittest.TestCase):

    # Test initialization of Tensors
    # ---------------------------------
    def test_shape_and_dim_withdim1(self):
        data = [1, 2, 4]
        t = nn.Tensor(data)
        self.assertEqual((3, ), t.shape)
        self.assertEqual(1, t.dim)

    def test_shape_and_dim_withdim2(self):
        data = [[1], [2]]
        t = nn.Tensor(data)
        self.assertEqual((2, 1), t.shape)
        self.assertEqual(2, t.dim)
        self.assertEqual(nn.Tensor, t[0].__class__)
    


    # Test operations
    # ---------------------------------

    # Mult
    def test_mult_vector_with_vector(self):
        a = nn.Tensor([1, 2])
        b = nn.Tensor([10, 20])
        c = a*b
        self.assertEqual([10, 40], c.data)

    def test_mult_vector_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([1])
        b = nn.Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a*b)
    
    def test_mult_2dmatrix_with_vector(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20])
        c = a*b
        self.assertEqual([10, 40], c[0].data)
        self.assertEqual([30, 80], c[1].data)
    
    def test_mult_2dmatrix_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a*b)

    def test_mult_vector_with_2dmatrix(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20])
        c = b*a
        self.assertEqual([10, 40], c[0].data)
        self.assertEqual([30, 80], c[1].data)
    def test_mult_matrix_with_matrix(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        res = a*a
        self.assertEqual([[1, 4], [9, 16]], res.tolist())
    def test_mult_scalar(self):
        v = nn.Tensor([6, 8])
        m = nn.Tensor([[12, 8], [16, 32]])
        self.assertEqual([12, 16], (v*2).data)
        self.assertEqual([[24, 16], [32, 64]], (m*2).tolist())
    
    # Add
    def test_add_vector_with_vector(self):
        a = nn.Tensor([1, 2])
        b = nn.Tensor([10, 20])
        self.assertEqual([11, 22], (a+b).data)

    def test_add_vector_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([1])
        b = nn.Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a+b)
    
    def test_add_2dmatrix_with_vector(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20])
        c = a+b
        self.assertEqual([11, 22], c[0].data)
        self.assertEqual([13, 24], c[1].data)
    
    def test_add_2dmatrix_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a+b)
    
    def test_add_mat_and_mat(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([[5, 6], [7, 8]])
        self.assertEqual([[6,8], [10, 12]], (a+b).tolist())
        self.assertEqual([[6,8], [10, 12]], (b+a).tolist())

    def test_add_vector_with_2dmatrix(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20])
        c = b+a
        self.assertEqual([11, 22], c[0].data)
        self.assertEqual([13, 24], c[1].data)

    def test_add_scalar(self):
        v = nn.Tensor([6, 8])
        m = nn.Tensor([[12, 8], [16, 32]])
        self.assertEqual([8, 10], (v+2).data)
        self.assertEqual([[14, 10], [18, 34]], (m+2).tolist())

    # Sub
    def test_sub_vector_with_vector(self):
        a = nn.Tensor([1, 2])
        b = nn.Tensor([10, 20])
        c = a-b
        self.assertEqual([-9, -18], c.data)
    
    def test_sub_vector_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([1])
        b = nn.Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a-b)

    def test_sub_2dmatrix_with_vector(self):
        m = nn.Tensor([[1, 2], [3, 4]])
        v = nn.Tensor([10, 20])
        c = m-v
        self.assertEqual([-9, -18], c[0].data)
        self.assertEqual([-7, -16], c[1].data)

    def test_sub_2dmatrix_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([[1, 2], [3, 4]])
        b = nn.Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a-b)

    def test_sub_vector_with_2dmatrix(self):
        m = nn.Tensor([[1, 2], [3, 4]])
        v = nn.Tensor([10, 20])
        c = v-m
        self.assertEqual([9, 18], c[0].data)
        self.assertEqual([7, 16], c[1].data)

    def test_sub_mat_and_mat(self):
        m = nn.Tensor([[1, 2], [3,4]])
        m1 = nn.Tensor([[5, 10], [20, 30]])
        self.assertEqual([[-4, -8], [-17, -26]], (m-m1).tolist())

    def test_sub_scalar(self):
        v = nn.Tensor([6, 8])
        m = nn.Tensor([[12, 8], [16, 32]])
        self.assertEqual([4, 6], (v-2).data)
        self.assertEqual([[10, 6], [14, 30]], (m-2).tolist())
    
    # Div
    def test_div_vector_with_vector(self):
        a = nn.Tensor([12, 8])
        b = nn.Tensor([2, 4])
        c = a/b
        self.assertEqual([6, 2], c.data)
    
    def test_div_vector_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([1])
        b = nn.Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a/b)

    def test_div_2d_with_vector(self):
        v = nn.Tensor([2, 4])
        m = nn.Tensor([[12, 8], [16, 32]])
        c = m/v
        self.assertEqual([6, 2], c[0].data)
        self.assertEqual([8, 8], c[1].data)

    def test_div_mat_and_mat(self):
        m1 = nn.Tensor([[1, 2], [5, 4]])
        m2 = nn.Tensor([[5, 4], [4, 16]])
        self.assertEqual([[0.2, 0.5], [1.25, 0.25]], (m1/m2).tolist())

    def test_div_vector_with_2d(self):
        v = nn.Tensor([6, 8])
        m = nn.Tensor([[12, 8], [16, 32]])
        c = v/m
        self.assertEqual([0.5, 1.0], c[0].data)
        self.assertEqual([0.375, 0.25], c[1].data)
    def test_div_scalar(self):
        v = nn.Tensor([6, 8])
        m = nn.Tensor([[12, 8], [16, 32]])
        self.assertEqual([3, 4], (v/2).data)
        self.assertEqual([[6, 4], [8, 16]], (m/2).tolist())
    
    # Dot
    def test_dot_vector_with_vector(self):
        a = nn.Tensor([1, 2])
        b = nn.Tensor([10, 20])
        self.assertEqual(50, (a@b).data)
        self.assertEqual(50, (b@a).data)
    
    def test_dot_vector_with_vector_shapes_mismatch_raises(self):
        a = nn.Tensor([1])
        b = nn.Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a-b)
    
    def test_dot_2dmatrix_with_vector(self):
        m = nn.Tensor([[1, 2], [3, 4]])
        v = nn.Tensor([10, 20])
        c = m@v
        self.assertEqual([50, 110], c.data)

    def test_dot_2dmatrix_with_vector_shapes_mismatch_raises(self):
        m = nn.Tensor([[1, 2], [3, 4]])
        v = nn.Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: m@v)
    
    def test_dot_vector_with_2dmatrix(self):
        # 3 * 3x2
        v = nn.Tensor([7, 2, 3])
        m = nn.Tensor([[1, 2], [3, 4], [5, 6]])
        c = v@m
        self.assertEqual([28, 40], c.data)
        # 3 * 3x3
        v = nn.Tensor([1, 2])
        m = nn.Tensor([[10, 20, 5], [30, 40, 20]])
        c = v@m
        self.assertEqual([70, 100, 45], c.data)
        # 3 * 3x4
        v = nn.Tensor([1, 2])
        m = nn.Tensor([[10, 20, 5, 30], [30, 40, 20, 15]])
        c = v@m
        self.assertEqual([70, 100, 45, 60], c.data)
    
    def test_dot_vector_with_2dmatrix_shapes_mismatch_raises(self):
        v = nn.Tensor([10, 20])
        m = nn.Tensor([[1, 2], [3, 4], [5,6]])
        self.assertRaises(RuntimeError, lambda: v@m)
    
    def test_dot_2d_with_2d(self):
        x = nn.Tensor([
            [7, 2, 3], 
            [6, 2, 7]
            ])
        m = nn.Tensor([[1, 2], [3, 4], [5,6]])
        c = x@m
        self.assertEqual([28, 40], c[0].data)
        self.assertEqual([47, 62], c[1].data)
    
    # exp(x)
    def test_exp_vector(self):
        v = nn.Tensor([-1, 0, 1, 2])
        res = v.exp()
        self.assertAlmostEqual(0.3678, res.data[0], delta=0.0001)
        self.assertAlmostEqual(1.0, res.data[1])
        self.assertAlmostEqual(math.e, res.data[2])
    
    def test_exp_matrix(self):
        m = nn.Tensor([[0, 1], [1, 0]])
        res = m.exp().tolist()
        self.assertEqual([[1, math.e], [math.e, 1]], res)

    # log(x)
    def test_log_vector(self):
        v = nn.Tensor([-1, 0, 1, 0.3678])
        res = v.log()
        self.assertEqual([math.nan, math.nan, 0], res.data[:3])
        self.assertAlmostEqual(-1, res.data[3], delta=0.01)

    def test_log_matrix(self):
        m = nn.Tensor([[1, math.e], [math.e, 1]])
        res = m.log().tolist()
        self.assertEqual([[0, 1], [1, 0]], res)
    
    # exp(x)
    def test_tanh_vector(self):
        v = nn.Tensor([-1, 0, 1])
        res = nn.tanh(v)
        self.assertAlmostEqual(-0.7616, res[0], delta=0.00001)
        self.assertAlmostEqual(0, res[1])
        self.assertAlmostEqual(0.7616, res[2], delta=0.00001)
    
    def test_tanh_matrix(self):
        v = nn.Tensor([[-1, 0, 1]])
        res = nn.tanh(v)
        res = res[0]
        self.assertAlmostEqual(-0.7616, res[0], delta=0.00001)
        self.assertAlmostEqual(0, res[1])
        self.assertAlmostEqual(0.7616, res[2], delta=0.00001)

    # Backprop
    # ---------------------------------
    # Add
    def test_grad_add_vec_and_vec(self):
        a = nn.Tensor([2, -2], requires_grad=True)
        b = nn.Tensor([1, -1], requires_grad=True)
        c = a+b; c.grad=1
        c.backward()
        self.assertEqual([1, 1], a.grad.data)
        self.assertEqual([1, 1], b.grad.data)

    def test_grad_add_mat_and_vec(self):
        a = nn.Tensor([[1,1], [2,2]], requires_grad=True)
        b = nn.Tensor([2, 2], requires_grad=True)
        c = a+b; c.grad=1
        c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.tolist())
        self.assertEqual([2,2], b.grad.data)
    
    def test_grad_add_vec_and_mat(self):
        a = nn.Tensor([[1,1], [2,2]], requires_grad=True)
        b = nn.Tensor([2, 2], requires_grad=True)
        c = b+a; c.grad=1; c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.tolist())
        self.assertEqual([2,2], b.grad.data)

    def test_grad_add_mat_and_mat(self):
        a = nn.Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = nn.Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a+b; c.grad=1; c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.tolist())
        self.assertEqual([[1,1], [1,1]], b.grad.tolist())

        a.grad=0; b.grad=0
        c = b+a; c.grad=1; c.backward()
        self.assertEqual([[1,1], [1,1]], a.grad.tolist())
        self.assertEqual([[1,1], [1,1]], b.grad.tolist())
    
    # Mult
    def test_grad_mult_vec_and_vec(self):
        a = nn.Tensor([2, 2], requires_grad=True)
        b = nn.Tensor([4, 4], requires_grad=True)
        c = a*b; c.grad=1
        c.backward()
        self.assertEqual([4, 4], a.grad.data)
        self.assertEqual([2, 2], b.grad.data)
    
    def test_grad_mult_mat_and_vec(self):
        a = nn.Tensor([[1,2], [3, 4]], requires_grad=True)
        b = nn.Tensor([5,6], requires_grad=True)
        c = a*b; c.grad=1; c.backward()

        self.assertEqual([[5, 6], [5, 6]], a.grad.tolist())
        self.assertEqual([4, 6], b.grad.data)
    
    def test_grad_mult_vec_and_mat(self):
        a = nn.Tensor([[1,2], [3, 4]], requires_grad=True)
        b = nn.Tensor([5,6], requires_grad=True)
        c = b*a; c.grad=1; c.backward()

        self.assertEqual([[5, 6], [5, 6]], a.grad.tolist())
        self.assertEqual([4, 6], b.grad.data)
    
    def test_grad_mult_mat_and_mat(self):
        a = nn.Tensor([[1,2], [3, 4]], requires_grad=True)
        b = nn.Tensor([[5,6], [7,8]], requires_grad=True)
        c = b*a; c.grad=1; c.backward()

        self.assertEqual([[5, 6], [7, 8]], a.grad.tolist())
        self.assertEqual([[1, 2], [3, 4]], b.grad.tolist())
    
    # Div
    def test_grad_div_vec_and_vec(self):
        x1 = nn.Tensor([2, 5, 5,   0.5], requires_grad=True)
        x2 = nn.Tensor([5, 2, 0.5, 5  ], requires_grad=True)
        c = x1/x2; c.grad=1; c.backward()

        self.assertEqual([0.2, 0.5, 2.0, 0.2], x1.grad.tolist())
        self.assertEqual([-0.08, -1.25, -20, -0.02], x2.grad.tolist())

    def test_grad_div_mat_and_vec(self):
        m = nn.Tensor([[5], [4]], requires_grad=True)
        v = nn.Tensor([2], requires_grad=True)
        c = m/v; c.grad=1; c.backward()

        self.assertEqual([[0.5], [0.5]], m.grad.tolist())
        self.assertEqual([-2.25], v.grad.tolist()) # also tests so that gradients are accumulated
    
    def test_grad_div_vec_and_mat(self):
        m = nn.Tensor([[5], [4]], requires_grad=True)
        v = nn.Tensor([2], requires_grad=True)
        c = v/m; c.grad=1; c.backward()

        self.assertEqual([[-0.08], [-0.125]], m.grad.tolist())
        self.assertEqual([0.45], v.grad.tolist()) 
    
    def test_grad_div_mat_and_mat(self):
        m1 = nn.Tensor([[2, 5], [5,   0.5]], requires_grad=True)
        m2 = nn.Tensor([[5, 2,], [0.5, 5]], requires_grad=True)
        c = m1/m2; c.grad=1; c.backward()

        self.assertEqual([[0.2, 0.5], [2.0, 0.2]], m1.grad.tolist())
        self.assertEqual([[-0.08, -1.25], [-20, -0.02]], m2.grad.tolist())
    
    ## Sub
    def test_grad_sub_vec_and_vec(self):
        x1 = nn.Tensor([-1, 0, 1], requires_grad=True)
        x2 = nn.Tensor([-1, 0, 1], requires_grad=True)
        c = x1-x2; c.grad=1; c.backward()

        self.assertEqual([1, 1, 1], x1.grad.tolist())
        self.assertEqual([-1, -1, -1], x2.grad.tolist())
    
    def test_grad_sub_mat_and_vec(self):
        m = nn.Tensor([[5], [5]], requires_grad=True)
        v = nn.Tensor([1], requires_grad=True)
        c = m-v; c.grad=1; c.backward()
        self.assertEqual([[1], [1]], m.grad.tolist())
        self.assertEqual([-2], v.grad.tolist())

    def test_grad_sub_mat_and_mat(self):
        m = nn.Tensor([[5], [5]], requires_grad=True)
        m1 = nn.Tensor([[5], [5]], requires_grad=True)
        c = m-m1; c.grad=1; c.backward()
        self.assertEqual([[1], [1]], m.grad.tolist())
        self.assertEqual([[-1], [-1]], m1.grad.tolist())


    # Other stuff, subscript, getitem, setitem etc
    # ---------------------------------
    def test_getitem_vector(self):
        m = nn.Tensor([1,2])
        self.assertEqual(1, m[0])
        self.assertEqual(2, m[1])

    def test_getitem_matrix(self):
        m = nn.Tensor([[1,2], [3,4]])
        self.assertEqual([1,2], m[0].data)
        self.assertEqual([3,4], m[1].data)

    def test_getmultipleitem_vector(self):
        m = nn.Tensor([1,2])
        self.assertEqual([2, 1], m[[1, 0]].data)

    def test_getmultipleitem_matrix(self):
        m = nn.Tensor([[1,2], [3,4]])
        self.assertEqual([[3,4], [1,2]], m[[1, 0]].tolist())

    def test_setitem_vector(self):
        v = nn.Tensor([1, 2])
        v[0] = 5
        self.assertEqual(5, v[0])

    def test_setitem_matrix(self):
        m = nn.Tensor([[1,2], [3,4]])
        m[0] = nn.Tensor([10, 20])
        self.assertEqual([10, 20], m[0].data)

    # Other functions
    # ---------------------------------
    def test_sum_vector(self):
        t = nn.Tensor([1, 2, 3])
        self.assertEqual(6, nn.sum(t))

    def test_sum_matrix(self):
        t = nn.Tensor([[1, 2, 3], [7, 3]])
        c = nn.sum(t)
        self.assertEqual([6, 10], c.data)
    
    def test_ones(self):
        a = nn.ones((3,))
        self.assertEqual([1, 1, 1], a.data)
        a = nn.ones((3, 5))
        self.assertEqual(3, len(a.data))
        self.assertEqual(5, len(a[0].data))

if __name__ == '__main__':
    unittest.main()