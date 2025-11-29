import numpy as np

def loss_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		return None
	if y.size <= 0 or y_hat.size <= 0:
		return None
	if y.ndim != 2 or y_hat.ndim != 2:
		return None
	if y.shape != y_hat.shape or y.shape[1] != 1 or y_hat.shape[1] != 1:
		return None
	
	m = y.shape[0]

	error = y_hat - y
	loss = (1/(m * 2)) * (error.T @ error)

	return float(loss[0][0])

X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

# Example 1:
print(loss_(X, Y))
# Output:
#2.142857142857143

# Example 2:
print(loss_(X, X))
# Output:
#0.0