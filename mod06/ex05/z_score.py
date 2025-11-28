import numpy as np

def zscore(x):
	if not isinstance(x, np.ndarray):
		return None
	if x.size <= 0 or x.ndim > 2:
		return None
	if x.ndim == 2 and x.shape[0] != 1 and x.shape[1] != 1:
		return None
	
	X_Prime = (x - x.mean()) / x.std(ddof=0)
	return X_Prime.flatten()

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(repr(zscore(X)))
# Output:
#array([-0.08620324, 1.2068453 , -0.86203236,
#0.17240647, -1.89647119])
#0.51721942,

# Example 2:
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(repr(zscore(Y)))
# Output:
#array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731,
#0.28795027, -1.72770165])