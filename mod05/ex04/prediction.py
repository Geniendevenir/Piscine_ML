import numpy as np

def predict_(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		return None
	if x.size == 0 or theta.size == 0:
		return None
	if x.ndim != 1:
		return None
	if theta.shape != (2, ) and theta.shape != (2, 1):
		return None

	# Add Intercept
	m = x.shape[0]
	X_prime = np.column_stack((np.ones((m,)), x))  # shape (m, 2)

	# Matrix multiplication
	y_hat = X_prime @ theta  # shape (m, 1)

	return y_hat


x = np.arange(1,6)
# Example 1:
theta1 = np.array([5, 0])
print(repr(predict_(x, theta1)))
# Ouput:
#array([5., 5., 5., 5., 5.])
# Do you understand why y_hat contains only 5s here?

# Example 2:
theta2 = np.array([0, 1])
print(repr(predict_(x, theta2)))
# Output:
#array([1., 2., 3., 4., 5.])
# Do you understand why y_hat == x here?

# Example 3:
theta3 = np.array([5, 3])
print(repr(predict_(x, theta3)))
# Output:
#array([ 8., 11., 14., 17., 20.])

# Example 4:
theta4 = np.array([-3, 1])
print(repr(predict_(x, theta4)))
# Output:
#array([-2., -1., 0., 1., 2.])