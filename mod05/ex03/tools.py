import numpy as np

def add_intercept(x):
	if not isinstance(x, np.ndarray) or x.size == 0: 
		return None

	if x.ndim == 1:
		x = x.reshape(-1, 1)

	m, n = x.shape
	ones = np.ones((m, 1))
	return np.concatenate([ones, x], axis=1)
	

# Example 1:
x = np.arange(1,6)
print(repr(add_intercept(x)))
# Output:
""" array([[1., 1.],
			[1., 2.],
			[1., 3.],
			[1., 4.],
			[1., 5.]]) """

# Example 2:
y = np.arange(1,10).reshape((3,3))
print(repr(add_intercept(y)))
# Output:
""" array([[1., 1., 2., 3.],
			[1., 4., 5., 6.],
			[1., 7., 8., 9.]]) """