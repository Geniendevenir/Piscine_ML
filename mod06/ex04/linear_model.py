import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR

#Parse CSV + Turn it into numpy arrays
df = pd.read_csv("are_blue_pills_magic.csv")
data = df.to_numpy()

x = data[..., 1].reshape(-1, 1)
y = data[..., 2].reshape(-1, 1)
thetas = np.array([[0], [1]])

#Linear Regression to get the optimal Thetas values
lr = MyLR(thetas, alpha=1e-3, max_iter=50000)
thetas_opt = lr.fit_(x, y)

#Predict and calculate loss with the optimal thetas
lr = MyLR(thetas_opt)
y_hat = lr.predict_(x)
loss = lr.loss_(y, y_hat)

#Plot Prediction vs Actual
blue_cyan = np.array((0, 255, 255)) / 255

plt.scatter(x, y, color=blue_cyan, label=r"$S_{\mathrm{true}}(\mathrm{pills})$")
plt.scatter(x, y_hat, c="#00FF00", marker="*", linewidths=1.5, label=r"$S_{\mathrm{predict}}(\mathrm{pills})$")
plt.plot(x, y_hat, c="#00FF00", ls="--")
plt.grid(True)
plt.xlabel("Quantity of blue pill (in micrograms)")
plt.ylabel("Space driving score")
plt.legend()
plt.tight_layout()
plt.show()


#Plot the loss function J(Theta) function of theta's
t0_opt, t1_opt = float(thetas_opt[0, 0]), float(thetas_opt[1, 0])
print(t0_opt, t1_opt)
theta0_vals = t0_opt + np.array([-40, -20, -10, 0, 10, 20], dtype=float)
theta1_vals = np.linspace(-13.8, -4, 400)

t0 = t0_opt - 5.5
for _ in range(6):
	costs = np.zeros_like(theta1_vals)

	for j, t1 in enumerate(theta1_vals):
		thetas = np.array([[t0], [t1]], dtype=float)
		lr = MyLR(thetas)
		y_hat = lr.predict_(x)
		costs[j] = lr.mse_(y, y_hat) / 2

	t0 += 2.5
	print(np.mean(costs))
	plt.xlim(-14.5, -3.5)
	plt.ylim(12, 148)
	plt.plot(theta1_vals, costs)

""" for t0 in theta0_vals:
	costs = np.zeros_like(theta1_vals)

	for j, t1 in enumerate(theta1_vals):
		thetas = np.array([[t0], [t1]], dtype=float)
		lr = MyLR(thetas)
		y_hat = lr.predict_(x)
		costs[j] = lr.mse_(y, y_hat)

	plt.plot(theta1_vals, costs) """

plt.grid(True)
plt.show()







