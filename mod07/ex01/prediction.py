import numpy as np

def predict_(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		return None
	if x.ndim != 2:
		return None
	if theta.shape != (x.shape[1] + 1, 1):
		return None

	m, n = x.shape
	X_Prime = np.hstack((np.ones((m,1)), x))

	y_hat = X_Prime @ theta
	return y_hat

x = np.arange(1,13).reshape((4,-1))
theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print(repr(predict_(x, theta1)))

# Example 2:
theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(repr(predict_(x, theta2)))
# Output:
#array([[ 1.], [ 4.], [ 7.], [10.]])
# Do you understand why y_hat == x[:,0] here?

# Example 3:
theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(repr(predict_(x, theta3)))
# Output:
#array([[ 9.64], [24.28], [38.92], [53.56]])

# Example 4:
theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(repr(predict_(x, theta4)))
# Output:
#array([[12.5], [32. ], [51.5], [71. ]])