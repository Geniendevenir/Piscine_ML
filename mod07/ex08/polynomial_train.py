import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from my_linear_regression import MyLinearRegression as MyLR
import pandas as pd

data = pd.read_csv("are_blue_pills_magics.csv")
X = np.array(data['Micrograms']).reshape(-1,1)
Y = np.array(data['Score']).reshape(-1,1)

# Create polynomial features for each degree
X1 = add_polynomial_features(X, 1)  # degree 1: [x]
X2 = add_polynomial_features(X, 2)  # degree 2: [x, x^2]
X3 = add_polynomial_features(X, 3)  # degree 3: [x, x^2, x^3]
X4 = add_polynomial_features(X, 4)  # degree 4: [x, x^2, x^3, x^4]
X5 = add_polynomial_features(X, 5)  # degree 5
X6 = add_polynomial_features(X, 6)  # degree 6

my_lr1 = MyLR(np.ones(2).reshape(-1,1), alpha=0.001, max_iter=1000000, normalize=True)
my_lr1.fit_(X1, Y)
Y_hat1 = my_lr1.predict_(X1)
mse1 = my_lr1.mse_(Y, Y_hat1)
print(f"Degree 1 MSE: {mse1}")


my_lr2 = MyLR(np.ones(3).reshape(-1,1), alpha=0.001, max_iter=1000000, normalize=True)
my_lr2.fit_(X2, Y)
Y_hat2 = my_lr2.predict_(X2)
mse2 = my_lr2.mse_(Y, Y_hat2)
print(f"Degree 2 MSE: {mse2}")

my_lr3 = MyLR(np.ones(4).reshape(-1,1), alpha=0.001, max_iter=1000000, normalize=True)
my_lr3.fit_(X3, Y)
Y_hat3 = my_lr3.predict_(X3)
mse3 = my_lr3.mse_(Y, Y_hat3)
print(f"Degree 3 MSE: {mse3}")

theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
my_lr4 = MyLR(theta4, alpha=1e-6, max_iter=1000000, normalize=False)
my_lr4.fit_(X4, Y)
Y_hat4 = my_lr4.predict_(X4)
mse4 = my_lr4.mse_(Y, Y_hat4)
print(f"Degree 4 MSE: {mse4}")

theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
my_lr5 = MyLR(theta5, alpha=1e-8, max_iter=1000000, normalize=False)
my_lr5.fit_(X5, Y)
Y_hat5 = my_lr5.predict_(X5)
mse5 = my_lr5.mse_(Y, Y_hat5)
print(f"Degree 5 MSE: {mse5}")

theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
my_lr6 = MyLR(theta6, alpha=1e-9, max_iter=1000000, normalize=False)
my_lr6.fit_(X6, Y)
Y_hat6 = my_lr6.predict_(X6)
mse6 = my_lr6.mse_(Y, Y_hat6)
print(f"Degree 6 MSE: {mse6}")

# Create bar plot of MSE scores
degrees = [1, 2, 3, 4, 5, 6]
mse_scores = [mse1, mse2, mse3, mse4, mse5, mse6]

plt.figure(figsize=(10, 6))
plt.bar(degrees, mse_scores, color='skyblue', edgecolor='black')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE Score')
plt.title('MSE Score vs Polynomial Degree')
plt.xticks(degrees)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Create smooth prediction curves
continuous_x = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

# Generate polynomial features for continuous x
continuous_x1 = add_polynomial_features(continuous_x, 1)
continuous_x2 = add_polynomial_features(continuous_x, 2)
continuous_x3 = add_polynomial_features(continuous_x, 3)
continuous_x4 = add_polynomial_features(continuous_x, 4)
continuous_x5 = add_polynomial_features(continuous_x, 5)
continuous_x6 = add_polynomial_features(continuous_x, 6)

# Generate predictions
y_hat1_smooth = my_lr1.predict_(continuous_x1)
y_hat2_smooth = my_lr2.predict_(continuous_x2)
y_hat3_smooth = my_lr3.predict_(continuous_x3)
y_hat4_smooth = my_lr4.predict_(continuous_x4)
y_hat5_smooth = my_lr5.predict_(continuous_x5)
y_hat6_smooth = my_lr6.predict_(continuous_x6)

# Plot all models and data points
plt.figure(figsize=(12, 8))
plt.scatter(X, Y, color='black', s=50, label='Data points', zorder=5)
plt.plot(continuous_x, y_hat1_smooth, label='Degree 1', linewidth=2)
plt.plot(continuous_x, y_hat2_smooth, label='Degree 2', linewidth=2)
plt.plot(continuous_x, y_hat3_smooth, label='Degree 3', linewidth=2)
plt.plot(continuous_x, y_hat4_smooth, label='Degree 4', linewidth=2)
plt.plot(continuous_x, y_hat5_smooth, label='Degree 5', linewidth=2)
plt.plot(continuous_x, y_hat6_smooth, label='Degree 6', linewidth=2)
plt.xlabel('Micrograms')
plt.ylabel('Score')
plt.title('Polynomial Models (Degree 1-6) vs Data')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(y_hat3_smooth)