import numpy as np

class MyLinearRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def predict_(self, x):
		if not isinstance(x, np.ndarray):
			return None
		if x.ndim != 2 or x.shape[1] != 1:
			return None
		m = x.shape[0]
		X_Prime = np.hstack((np.ones((m, 1)), x))
		y_hat = X_Prime @ self.thetas
		return y_hat


	def loss_elem_(self, y, y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			return None
		if y.ndim != 2 or y_hat.ndim != 2:
			return None
		if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape != y_hat.shape:
			return None
		
		return pow(y_hat - y, 2)

	def loss_(self, y ,y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			return None
		if y.ndim != 2 or y_hat.ndim != 2:
			return None
		if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape != y_hat.shape:
			return None

		error = self.loss_elem_(y, y_hat)

		m = y.shape[0]
		return float((1/(m*2)) * np.sum(error))

	def mse_(self, y, y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			return None
		if y.ndim != 2 or y_hat.ndim != 2:
			return None
		if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape != y_hat.shape:
			return None

		m = y.shape[0]
		error = y_hat - y
		return float((1 / m) * np.sum(error ** 2))

	def fit_(self, x, y):
		if not isinstance(x, np.ndarray):
			return None
		if x.ndim != 2 or x.shape[1] != 1:
			return None
		if not isinstance(y, np.ndarray):
			return None
		if y.ndim != 2 or y.shape[1] != 1:
			return None
		if y.shape != x.shape:
			return None
		
		self.thetas = self.thetas.astype(float)

		for _ in range(self.max_iter):
			m = y.shape[0]
			X_Prime = np.hstack((np.ones((m, 1)), x))
			y_hat = self.predict_(x)

			gradient = (1/m) * X_Prime.T @ (y_hat - y)
			self.thetas -= self.alpha * gradient
		return self.thetas



