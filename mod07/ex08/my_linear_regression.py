import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MyLinearRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000, normalize=True):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		self.normalize = normalize
		self.x_min = None
		self.x_max = None

	def minmax(self, x):
		if not isinstance(x, np.ndarray):
			return None
		if x.size <= 0 or x.ndim != 2:
			return None
		X_Prime = (x - x.min()) / (x.max() - x.min()).flatten()
		return X_Prime

	def predict_(self, x):
		if not isinstance(x, np.ndarray):
			return None
		if x.size <= 0 or x.ndim != 2:
			return None
	
		# Apply same normalization as during training if normalize is True
		if self.normalize and self.x_min is not None and self.x_max is not None:
			x_normalized = (x - self.x_min) / (self.x_max - self.x_min)
		else:
			x_normalized = x
		
		m = x.shape[0]
		X_Prime = np.hstack((np.ones((m, 1)), x_normalized))
		y_hat = X_Prime @ self.thetas
		return y_hat


	def loss_elem_(self, y, y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			return None
		if y_hat.ndim == 1:
			y_hat = y_hat.reshape(-1, 1)
		if y.ndim != 2 or y_hat.ndim != 2:
			return None
		if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape != y_hat.shape:
			return None
		
		return pow(y_hat - y, 2)

	def loss_(self, y ,y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			return None
		if y_hat.ndim == 1:
			y_hat = y_hat.reshape(-1, 1)
		if y.ndim != 2 or y_hat.ndim != 2:
			return None
		if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape != y_hat.shape:
			return None

		error = self.loss_elem_(y, y_hat)

		m = y.shape[0]
		return float((1 / (m * 2)) * np.sum(error))

	def mse_(self, y, y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			return None
		if y_hat.ndim == 1:
			y_hat = y_hat.reshape(-1, 1)
		if y.ndim != 2 or y_hat.ndim != 2:
			return None
		if y.shape[1] != 1 or y_hat.shape[1] != 1 or y.shape != y_hat.shape:
			return None

		m = y.shape[0]
		error = y_hat - y
		return float((1 / m) * (error.T @ error))

	def fit_(self, x, y):
		if not isinstance(x, np.ndarray):
			return None
		if x.size <= 0 or x.ndim != 2:
			return None
		if not isinstance(y, np.ndarray):
			return None
		if y.size <= 0 or y.ndim != 2 or y.shape[1] != 1:
			return None
		if x.shape[0] != y.shape[0]:
			return None
		if not isinstance(self.thetas, np.ndarray):
			self.thetas = np.array(self.thetas).reshape(-1, 1)
		
		self.thetas = self.thetas.astype(float)
		
		# Store normalization parameters and normalize input if normalize is True
		if self.normalize:
			self.x_min = x.min(axis=0, keepdims=True)
			self.x_max = x.max(axis=0, keepdims=True)
			x_normalized = (x - self.x_min) / (self.x_max - self.x_min)
		else:
			x_normalized = x

		for _ in range(self.max_iter):
			m = y.shape[0]

			X_Prime = np.hstack((np.ones((m, 1)), x_normalized))
			y_hat = X_Prime @ self.thetas
			error = y_hat - y

			gradient = (1 / m) * (X_Prime.T @ error)
			self.thetas -= self.alpha * gradient

		return self.thetas
