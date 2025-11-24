import numpy as np

#ŷ(i) = θ0 + θ1 x(i)

def simple_predict(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		return None
	elif x.size == 0 or theta.size == 0:
		return None
	elif len(x.shape) != 1 or (len(theta.shape) != 1 and theta.shape[0] != 2):
		return None
	return (theta[0] + x * theta[1]).astype(float)

x = np.arange(1,6)
# Example 1:
theta1 = np.array([5, 0])
print(repr(simple_predict(x, theta1)))
# Ouput:
#array([5., 5., 5., 5., 5.])
# Do you understand why y_hat contains only 5s here?

# Example 2:
theta2 = np.array([0, 1])
print(repr(simple_predict(x, theta2)))
# Output:
#array([1., 2., 3., 4., 5.])
# Do you understand why y_hat == x here?

# Example 3:
theta3 = np.array([5, 3])
print(repr(simple_predict(x, theta3)))
# Output:
#array([ 8., 11., 14., 17., 20.])

# Example 4:
theta4 = np.array([-3, 1])
print(repr(simple_predict(x, theta4)))
# Output:
#array([-2., -1., 0., 1., 2.])