import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

def cost(theta0, theta1, x, y):
    m = len(x)
    y_hat = theta0 + theta1 * x
    return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)


df = pd.read_csv("are_blue_pills_magic.csv")
data = df.to_numpy()

x = data[..., 1].reshape(-1, 1)
y = data[..., 2].reshape(-1, 1)

theta_0 = 0

theta1_vals = np.linspace(-3, 20, 400)
cost_vals_theta1 = np.array([
    cost(theta_0, t1, x, y) for t1 in theta1_vals
])
print(repr(cost_vals_theta1))
plt.plot(theta1_vals, cost_vals_theta1)


""" theta_1 = 0
theta0_vals = np.linspace(-3, 7, 400)
cost_vals_theta0 = np.array([
    cost(t0, theta_1, x, y) for t0 in theta0_vals
])

plt.plot(theta0_vals, cost_vals_theta0) """

plt.show()