class Vector:
	def __init__(self, value):
		#Instantiate Column Vector with a size (ex: Vector(3) = [[0.0], [1.0], [2.0], [3.0]]
		if isinstance(value, int) and value > 0:
			self.value = [[float(i)] for i in range(value)]
			self.shape = (value, 1)

		#Instantiate Row Vector with a range (ex: Vector((10,16)) = [[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]
		elif isinstance(value, tuple) and len(value) == 2:
			if not isinstance(value[0], int) or not isinstance(value[1], int):
				return print("Error: Invalid Value")
			elif value[0] < 0 or value[1] < 0:
				return print("Error: Invalid Value")
			elif value[0] > value[1]:
				return print("Error: Range Constructor((a, b): a must be < b")
			self.value = [[float(i + value[0])] for i in range(value[1] - value[0])]
			self.shape = (1, value[1] - value[0])

		#Instantiate with a Column/Row Vector
		elif not isinstance(value, list):
			return print("Error: Invalid Value")

		#Potential Column Vector
		elif all(isinstance(element, list) and len(element) == 1 and isinstance(element[0], float) for element in value):
			if len(value) < 2:
				return print("Error: Invalid Value")
			self.value = value
			self.shape = (len(value), 1)

		#Potential Row Vector
		elif not len(value) == 1 or not isinstance(value[0], list) or not len(value[0]) > 1:
			return print("Error: Invalid Value")
		elif all (isinstance(element, float) for element in value[0]):
			self.value = value
			self.shape = (1, len(value[0]))
		else:
			return print("Error: Invalid Value")

	def dot(self, v2):
		#Produces a dot product between two vectors of same shape
		#print(f"Shape self = {self.shape} | Shape v2 = {v2.shape}")
		if isinstance(v2, Vector) and self.shape == v2.shape:
			if self.shape[0] == 1:
				return sum(a * b for a, b in zip(self.value[0], v2.value[0]))
			elif self.shape[1] == 1:
				return sum(a * b for elemA, elemB in zip(self.value, v2.value) for a, b in zip(elemA, elemB))
		else:
			return print("Error Dot Product")
			"""raise TypeError("Dot Product Error")"""

	def T(self):
		#returns the transpose vector
		if self.shape[0] == 1: #row to column
			return Vector([[x] for x in self.value[0]]) 
		elif self.shape[1] == 1: #column to row
			return Vector([[nbr[0] for nbr in self.value]])

	# add & radd : only vectors of the same shape.
	def __add__(self, v2):
		#+Scalar
		if isinstance(v2, int):
			if self.shape[0] == 1:
				return Vector([[a + v2 for a in self.value[0]]])
			elif self.shape[1] == 1:
				return Vector([[a + v2] for elemA in self.value for a in elemA])

		#+Vector
		elif isinstance(v2, Vector) and self.shape == v2.shape:
			if self.shape[0] == 1:
				return Vector([[a + b for a, b in zip(self.value[0], v2.value[0])]])
			elif self.shape[1] == 1:
				return Vector([[a + b] for elemA, elemB in zip(self.value, v2.value) for a, b in zip(elemA, elemB)])
		else:
			return print("Error Add Vector")

	def __radd__(self, v2):
		return self.__add__(v2)

	# sub & rsub: only vectors of the same shape.
	def __sub__(self, v2):
		#-Scalar
		if isinstance(v2, int):
			if self.shape[0] == 1:
				return Vector([[a - v2 for a in self.value[0]]])
			elif self.shape[1] == 1:
				return Vector([[a - v2] for elemA in self.value for a in elemA])

		#-Vector
		elif isinstance(v2, Vector) and self.shape == v2.shape:
			if self.shape[0] == 1:
				return Vector([[a - b for a, b in zip(self.value[0], v2.value[0])]])
			elif self.shape[1] == 1:
				return Vector([[a - b] for elemA, elemB in zip(self.value, v2.value) for a, b in zip(elemA, elemB)])
		else:
			return print("Error Sub Vector")

	def __rsub__(self, v2):
		return self.__sub__(v2)

	def __truediv__(self, v2):
		#-Scalar
		if isinstance(v2, int) and v2 > 0:
			if self.shape[0] == 1:
				return Vector([[a / v2 for a in self.value[0]]])
			elif self.shape[1] == 1:
				return Vector([[a / v2] for elemA in self.value for a in elemA])

		else:
			return print("Error TrueDiv Vector")

	def __rtruediv__(self, v2):
		raise NotImplementedError("Division of a scalar by a Vector is not defined here.")

	# mul & rmul: only scalars (to perform multiplication of a Vector by a scalar).
	def __mul__(self, v2):
		if isinstance(v2, int):
			if self.shape[0] == 1:
				return Vector([[a * v2 for a in self.value[0]]])
			elif self.shape[1] == 1:
				return Vector([[a * v2] for elemA in self.value for a in elemA])

		else:
			return print("Error Mul Vector")

	def __rmul__(self, v2):
		return self.__mul__(v2)

	# must be identical, i.e we expect that print(vector) and vector within python interpretor to behave the same, see corresp
	def __str__(self):
		return f"{self.value}"

	def __repr__(self):
		return f"{self.value}"
