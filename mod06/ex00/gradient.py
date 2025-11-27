import numpy as np

def simple_gradient(x, y, theta):
	if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
		return None
	if y.ndim != 2 or x.ndim != 2:
		return None
	if y.shape[0] != x.shape[0]:
		return None
	if y.shape[1] != 1 or x.shape[1] != 1:
		return None
	if not isinstance(theta, np.ndarray):
		return None
	if theta.ndim != 2 or theta.shape != (2, 1):
		return None

	#Prediction h(xi)
	m = x.shape[0]
	X_prime = np.hstack((np.ones((m,1)), x))  # (m,2)
	y_hat = X_prime @ theta  # shape (m, 1)

	m = y.shape[0]
	error = y_hat - y

	#Gradient
	gradient_T0 = (1 / m) * np.sum(error)
	gradient_T1 = (1 / m) * np.sum(error * x)

	return np.array([[gradient_T0], [gradient_T1]])


x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

# Example 0:
theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(repr(simple_gradient(x, y, theta1)))
# Output:
#array([[-19.0342574], [-586.66875564]])

# Example 1:
theta2 = np.array([1, -0.4]).reshape((-1, 1))
print(repr(simple_gradient(x, y, theta2)))
# Output:
#array([[-57.86823748], [-2230.12297889]])