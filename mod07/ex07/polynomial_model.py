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

x = np.arange(1,6).reshape(-1, 1)
#Example 0:
print(repr(add_polynomial_features(x, 3)))
""" array([[ 1,  1,   1],
       [ 2,  4,   8],
       [ 3,  9,  27],
       [ 4, 16,  64],
       [ 5, 25, 125]]) """

#Example 1:
print(repr(add_polynomial_features(x, 6)))
""" array([[   1,    1,    1,    1,     1,	1],
       [   2,    4,    8,   16,    32,		64	],
       [   3,    9,   27,   81,   243,		729	],
       [   4,   16,   64,  256,  1024,		4096	],
       [   5,   25,  125,  625,  3125,		15625	]]) """