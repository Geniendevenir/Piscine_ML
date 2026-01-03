import numpy as np

def add_polynomial_features(x, power):
	if not isinstance(x, np.ndarray):
		return None
	if x.size <= 0 or x.ndim != 2:
		return None
	if x.shape[1] != 1:
		return None
	if not isinstance(power, int) or power <= 0:
		return None
	X_Prime = x.copy()
	for j in range(power):
		if j > 0:
			X_Prime = np.hstack((X_Prime, np.power(x, j + 1)))
	return X_Prime
