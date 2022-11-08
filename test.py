import unittest
import nn

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
    
    def test_ones(self):
        a = nn.ones((3,))
        self.assertEqual([1, 1, 1], a.data)
        a = nn.ones((3, 5))
        self.assertEqual(3, len(a.data))
        self.assertEqual(5, len(a[0].data))

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
        c = a@b
        self.assertEqual(50, c)
        self.assertEqual(a@b, b@a)
    
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

    # Other stuff
    # ---------------------------------
    def test_sum_vector(self):
        t = nn.Tensor([1, 2, 3])
        self.assertEqual(6, nn.sum(t))

    def test_sum_matrix(self):
        t = nn.Tensor([[1, 2, 3], [7, 3]])
        c = nn.sum(t)
        self.assertEqual([6, 10], c.data)

if __name__ == '__main__':
    unittest.main()