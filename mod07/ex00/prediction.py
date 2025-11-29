import numpy as np

def simple_predict(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		return None
	if x.ndim != 2:
		return None
	if theta.shape != (x.shape[1] + 1, 1):
		return None

	y_hat = np.zeros((x.shape[0], 1))

	for m in range(x.shape[0]):
		pred_i = theta[0, 0]
		for n in range(x.shape[1]):
			pred_i += theta[n + 1, 0] * x[m, n]
		y_hat[m, 0] = pred_i
	return y_hat

x = np.arange(1,13).reshape((4,-1))
theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print(repr(simple_predict(x, theta1)))

# Example 2:
theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(repr(simple_predict(x, theta2)))
# Output:
#array([[ 1.], [ 4.], [ 7.], [10.]])
# Do you understand why y_hat == x[:,0] here?

# Example 3:
theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(repr(simple_predict(x, theta3)))
# Output:
#array([[ 9.64], [24.28], [38.92], [53.56]])

# Example 4:
theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(repr(simple_predict(x, theta4)))
# Output:
#array([[12.5], [32. ], [51.5], [71. ]])