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
	
	def __repr__(self):
		return f'Tensor(data={self.data.__repr__()})'


data = [
	[[], [3, 5]]
]
a = Tensor(data)

print(a)
print('shape=',a.shape)
print('dim=',a.dim)