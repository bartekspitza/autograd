class Tensor:
	def __init__(self, data):
		self.data = data
		
		# Compute shape
		self.shape = ()
		curr_dim = data
		while True:
			if isinstance(curr_dim, list):
				curr_dim_len = len(curr_dim)
				self.shape = self.shape + (curr_dim_len, )

				if curr_dim_len > 0:
					curr_dim = curr_dim[0]
				else:
					break
			else:
				break
		self.dim = len(self.shape)
	

	# Ops
	def mult(self, other):
		if not isinstance(other, Tensor):
			raise TypeError("Not Tensor")
		
		# Vector x Vector
		if self.dim == 1 and other.dim == 1:
			if self.shape != other.shape:
				raise RuntimeError(f'Shape {self.shape} does not match {other.shape}')

			new_data = []	
			for a, b in zip(self.data, other.data):
				new_data.append(a*b)
			return Tensor(new_data)
		
		# Matrix(dim=2) x Vector
		if self.dim == 2 and other.dim == 1:
			new_data = []
			for d in self.data:
				tmp = Tensor(d)
				res = tmp.mult(other)
				new_data.append(res.data)
			return Tensor(new_data)

	
	def __repr__(self):
		return f'Tensor(data={self.data.__repr__()})'