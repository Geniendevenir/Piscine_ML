import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MyLinearRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

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
	
		m = x.shape[0]
		X_Prime = np.hstack((np.ones((m, 1)), x))
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
		for _ in range(self.max_iter):
			m = y.shape[0]

			X_Prime = np.hstack((np.ones((m, 1)), x))
			y_hat = self.predict_(x)
			error = y_hat - y

			gradient = (1 / m) * (X_Prime.T @ error)
			self.thetas -= self.alpha * gradient

		return self.thetas


data = pd.read_csv("spacecraft_data.csv")

x = np.array(data[['Age','Thrust_power','Terameters']])
y = np.array(data[['Sell_price']])

#UNIVARIATE MODEL

#UNIVARIATE MODEL 1: Price(Age)
thetas_age = np.array([[1], [2]])
myLR_age = MyLinearRegression(thetas_age, max_iter=50000)
x_noramlized = myLR_age.minmax(x)
x_age = x[..., 0].reshape(-1, 1)


myLR_age.fit_(x_age, y)
y_hat_age = myLR_age.predict_(x_age)

plt.scatter(x_age, y, color="#0000ff")
plt.scatter(x_age, y_hat_age, color="#5f84ff", s=10)
plt.xlabel("x1: age (in years)")
plt.ylabel("y: sell price (in keuros)")
plt.grid(True)
plt.show()


#UNIVARIATE MODEL 2: Price(Thrust)
thetas_thrust = np.array([[1], [2]])
myLR_thrust = MyLinearRegression(thetas_thrust, max_iter=200000)
x_thrust = x[..., 1].reshape(-1, 1)
x_thrust_n = x_noramlized[..., 1].reshape(-1, 1)

myLR_thrust.fit_(x_thrust_n, y)
y_hat_thrust = myLR_thrust.predict_(x_thrust_n)

plt.scatter(x_thrust, y, color="#005c1c")
plt.scatter(x_thrust, y_hat_thrust, color="#00ff00", s=10)
plt.xlabel("x2: thrust power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.grid(True)
plt.show()

#UNIVARIATE MODEL 3: Price(Distance)
thetas_distance = np.array([[1], [2]])
myLR_distance = MyLinearRegression(thetas_distance, max_iter=100000)
x_distance = x[..., 2].reshape(-1, 1)
x_distance_n = x_noramlized[..., 2].reshape(-1, 1)

myLR_distance.fit_(x_distance_n, y)
y_hat_distance = myLR_distance.predict_(x_distance_n)

plt.scatter(x_distance, y, color="#a521ce")
plt.scatter(x_distance, y_hat_distance, color="#fd70da", s=10)
plt.xlabel("x2: distance power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.grid(True)
plt.show()


""" #MULTIVARIATE MODEL
myLR = MyLinearRegression(thetas=[1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)
x_noramlized = myLR.minmax(x)
x_age = x[..., 0].reshape(-1, 1)
x_thrust = x[..., 1].reshape(-1, 1)
x_distance = x[..., 2].reshape(-1, 1)

myLR.fit_(x, y)
y_hat = myLR.predict_(x)

plt.scatter(x_age, y, color="#0000ff")
plt.scatter(x_age, y_hat, color="#5f84ff", s=10)
plt.xlabel("x1: age (in years)")
plt.ylabel("y: sell price (in keuros)")
plt.grid(True)
plt.show()

plt.scatter(x_thrust, y, color="#005c1c")
plt.scatter(x_thrust, y_hat, color="#00ff00", s=10)
plt.xlabel("x2: thrust power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.grid(True)
plt.show()

plt.scatter(x_distance, y, color="#a521ce")
plt.scatter(x_distance, y_hat, color="#fd70da", s=10)
plt.xlabel("x2: distance power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.grid(True)
plt.show() """