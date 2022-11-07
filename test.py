import unittest
from nn import Tensor

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

	def test_shape_and_dim_withdim3(self):
		data = [
			[[1], [2]], 
			[[3], [4]], 
			[[5], [6]]
			]
		t = Tensor(data)
		self.assertEqual((3, 2, 1), t.shape)
		self.assertEqual(3, t.dim)

	# Test operations
	# ---------------------------------

	# Mult
	def test_mult_vector_with_vector(self):
		a = Tensor([1, 2])
		b = Tensor([10, 20])
		c = a.mult(b)
		self.assertEqual([10, 40], c.data)

	def test_mult_vector_with_vector_shapes_mismatch_raises(self):
		a = Tensor([1])
		b = Tensor([10, 20])
		self.assertRaises(RuntimeError, lambda: a.mult(b))
	
	def test_mult_2dmatrix_with_vector(self):
		a = Tensor([[1, 2], [3, 4]])
		b = Tensor([10, 20])
		c = a.mult(b)
		self.assertEqual([10, 40], c.data[0])
		self.assertEqual([30, 80], c.data[1])
	
	def test_mult_2dmatrix_with_vector_shapes_mismatch_raises(self):
		a = Tensor([[1, 2], [3, 4]])
		b = Tensor([10, 20, 30])
		self.assertRaises(RuntimeError, lambda: a.mult(b))

	def test_mult_vector_with_2dmatrix(self):
		a = Tensor([[1, 2], [3, 4]])
		b = Tensor([10, 20])
		c = b.mult(a)
		self.assertEqual([10, 40], c.data[0])
		self.assertEqual([30, 80], c.data[1])
    
	# Add
	def test_add_vector_with_vector(self):
		a = Tensor([1, 2])
		b = Tensor([10, 20])
		c = a.add(b)
		self.assertEqual([11, 22], c.data)

	def test_add_vector_with_vector_shapes_mismatch_raises(self):
		a = Tensor([1])
		b = Tensor([10, 20])
		self.assertRaises(RuntimeError, lambda: a.add(b))
	
	def test_add_2dmatrix_with_vector(self):
		a = Tensor([[1, 2], [3, 4]])
		b = Tensor([10, 20])
		c = a.add(b)
		self.assertEqual([11, 22], c.data[0])
		self.assertEqual([13, 24], c.data[1])
	
	def test_add_2dmatrix_with_vector_shapes_mismatch_raises(self):
		a = Tensor([[1, 2], [3, 4]])
		b = Tensor([10, 20, 30])
		self.assertRaises(RuntimeError, lambda: a.add(b))

	def test_add_vector_with_2dmatrix(self):
		a = Tensor([[1, 2], [3, 4]])
		b = Tensor([10, 20])
		c = b.add(a)
		self.assertEqual([11, 22], c.data[0])
		self.assertEqual([13, 24], c.data[1])


if __name__ == '__main__':
	unittest.main()