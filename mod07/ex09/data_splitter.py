import numpy as np


def data_spliter(x, y, proportion):
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
		return None
	if x.size == 0 or y.size == 0 or x.ndim != 2 or  y.ndim != 2:
		return None
	if y.shape[1] != 1 or x.shape[0] != y.shape[0]:
		return None
	if not isinstance(proportion, float) or proportion <= 0 or proportion >= 1:
		return None


	rng = np.random.default_rng(42)	
	indices = np.arange(x.shape[0])
	rng.shuffle(indices)
	X_shuffled = x[indices]
	Y_shuffled = y[indices]

	trainingSize = int(x.shape[0] * proportion)

	X_training = X_shuffled[:trainingSize]
	Y_training = Y_shuffled[:trainingSize]
	X_test = X_shuffled[trainingSize:]
	Y_test = Y_shuffled[trainingSize:]

	return (X_training, X_test, Y_training, Y_test)
	


x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

# Example 1:
print(data_spliter(x1, y, 0.8))
# Output:
""" (array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1])) """

# Example 2:
print(data_spliter(x1, y, 0.5))
# Output:
""" (array([59, 10]), array([
1, 300,
42]), array([0, 1]), array([0, 0, 1])) """


x2 = np.array([[ 1, 42],
[300, 10],
[ 59, 1],
[300, 59],
[ 10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

# Example 3:
print(data_spliter(x2, y, 0.8))
# Output:
""" (array([[ 10, 42],
[300, 59],
[ 59,
1],
[300, 10]]),
array([[ 1, 42]]),
array([0, 1, 0, 1]),
array([0])) """

# Example 4:
print(data_spliter(x2, y, 0.5))
# Output:
""" (array([[59, 1],
[10, 42]]),
array([[300, 10],
[300, 59],
[ 1, 42]]),
array([0, 0]),
array([1, 1, 0])) """