import unittest
from nn import Tensor, ones

class Testing(unittest.TestCase):

    # Test initialization of Tensors
    # ---------------------------------
    def test_shape_and_dim_withdim1(self):
        data = [1, 2, 4]
        t = Tensor(data)
        self.assertEqual((3, ), t.shape)
        self.assertEqual(1, t.dim)

    def test_shape_and_dim_withdim2(self):
        data = [[1], [2]]
        t = Tensor(data)
        self.assertEqual((2, 1), t.shape)
        self.assertEqual(2, t.dim)
        self.assertEqual(Tensor, t[0].__class__)
    
    def test_ones(self):
        a = ones((3,))
        self.assertEqual([1, 1, 1], a.data)
        a = ones((3, 5))
        self.assertEqual(3, len(a.data))
        self.assertEqual(5, len(a[0].data))

    # Test operations
    # ---------------------------------

    # Mult
    def test_mult_vector_with_vector(self):
        a = Tensor([1, 2])
        b = Tensor([10, 20])
        c = a*b
        self.assertEqual([10, 40], c.data)

    def test_mult_vector_with_vector_shapes_mismatch_raises(self):
        a = Tensor([1])
        b = Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a*b)
    
    def test_mult_2dmatrix_with_vector(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20])
        c = a*b
        self.assertEqual([10, 40], c[0].data)
        self.assertEqual([30, 80], c[1].data)
    
    def test_mult_2dmatrix_with_vector_shapes_mismatch_raises(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a*b)

    def test_mult_vector_with_2dmatrix(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20])
        c = b*a
        self.assertEqual([10, 40], c[0].data)
        self.assertEqual([30, 80], c[1].data)
    
    # Add
    def test_add_vector_with_vector(self):
        a = Tensor([1, 2])
        b = Tensor([10, 20])
        self.assertEqual([11, 22], (a+b).data)

    def test_add_vector_with_vector_shapes_mismatch_raises(self):
        a = Tensor([1])
        b = Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a+b)
    
    def test_add_2dmatrix_with_vector(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20])
        c = a+b
        self.assertEqual([11, 22], c[0].data)
        self.assertEqual([13, 24], c[1].data)
    
    def test_add_2dmatrix_with_vector_shapes_mismatch_raises(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a+b)

    def test_add_vector_with_2dmatrix(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20])
        c = b+a
        self.assertEqual([11, 22], c[0].data)
        self.assertEqual([13, 24], c[1].data)
    
    # Sub
    def test_sub_vector_with_vector(self):
        a = Tensor([1, 2])
        b = Tensor([10, 20])
        c = a-b
        self.assertEqual([-9, -18], c.data)
    
    def test_sub_vector_with_vector_shapes_mismatch_raises(self):
        a = Tensor([1])
        b = Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a-b)

    def test_sub_2dmatrix_with_vector(self):
        m = Tensor([[1, 2], [3, 4]])
        v = Tensor([10, 20])
        c = m-v
        self.assertEqual([-9, -18], c[0].data)
        self.assertEqual([-7, -16], c[1].data)

    def test_sub_2dmatrix_with_vector_shapes_mismatch_raises(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: a-b)

    def test_sub_vector_with_2dmatrix(self):
        m = Tensor([[1, 2], [3, 4]])
        v = Tensor([10, 20])
        c = v-m
        self.assertEqual([9, 18], c[0].data)
        self.assertEqual([7, 16], c[1].data)
    
    # Div
    def test_div_vector_with_vector(self):
        a = Tensor([12, 8])
        b = Tensor([2, 4])
        c = a/b
        self.assertEqual([6, 2], c.data)
    
    def test_div_vector_with_vector_shapes_mismatch_raises(self):
        a = Tensor([1])
        b = Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a/b)

    def test_div_2d_with_vector(self):
        v = Tensor([2, 4])
        m = Tensor([[12, 8], [16, 32]])
        c = m/v
        self.assertEqual([6, 2], c[0].data)
        self.assertEqual([8, 8], c[1].data)

    def test_div_vector_with_2d(self):
        v = Tensor([6, 8])
        m = Tensor([[12, 8], [16, 32]])
        c = v/m
        self.assertEqual([0.5, 1.0], c[0].data)
        self.assertEqual([0.375, 0.25], c[1].data)
    
    # Dot
    def test_dot_vector_with_vector(self):
        a = Tensor([1, 2])
        b = Tensor([10, 20])
        c = a@b
        self.assertEqual(50, c)
        self.assertEqual(a@b, b@a)
    
    def test_dot_vector_with_vector_shapes_mismatch_raises(self):
        a = Tensor([1])
        b = Tensor([10, 20])
        self.assertRaises(RuntimeError, lambda: a-b)
    
    def test_dot_2dmatrix_with_vector(self):
        m = Tensor([[1, 2], [3, 4]])
        v = Tensor([10, 20])
        c = m@v
        self.assertEqual([50, 110], c.data)

    def test_dot_2dmatrix_with_vector_shapes_mismatch_raises(self):
        m = Tensor([[1, 2], [3, 4]])
        v = Tensor([10, 20, 30])
        self.assertRaises(RuntimeError, lambda: m@v)
    
    def test_dot_vector_with_2dmatrix(self):
        # 3 * 3x2
        v = Tensor([7, 2, 3])
        m = Tensor([[1, 2], [3, 4], [5, 6]])
        c = v@m
        self.assertEqual([28, 40], c.data)
        # 3 * 3x3
        v = Tensor([1, 2])
        m = Tensor([[10, 20, 5], [30, 40, 20]])
        c = v@m
        self.assertEqual([70, 100, 45], c.data)
        # 3 * 3x4
        v = Tensor([1, 2])
        m = Tensor([[10, 20, 5, 30], [30, 40, 20, 15]])
        c = v@m
        self.assertEqual([70, 100, 45, 60], c.data)
    
    def test_dot_vector_with_2dmatrix_shapes_mismatch_raises(self):
        v = Tensor([10, 20])
        m = Tensor([[1, 2], [3, 4], [5,6]])
        self.assertRaises(RuntimeError, lambda: v@m)
    
    def test_dot_2d_with_2d(self):
        x = Tensor([
            [7, 2, 3], 
            [6, 2, 7]
            ])
        m = Tensor([[1, 2], [3, 4], [5,6]])
        c = x@m
        self.assertEqual([28, 40], c[0].data)
        self.assertEqual([47, 62], c[1].data)

if __name__ == '__main__':
    unittest.main()