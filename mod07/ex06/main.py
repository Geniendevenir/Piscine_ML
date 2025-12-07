import pandas as pd
import numpy as np
from multivariate_linear_model import MyLinearRegression as MyLR

data = pd.read_csv("spacecraft_data.csv")

""" #UNIVARIATE TEST
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])

myLR_age = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)

myLR_age.fit_(X, Y)
y_pred = myLR_age.predict_(X)
print(myLR_age.mse_(Y, y_pred))
#Output
#55736.867198... """

#MULTIVARIATE TEST
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])
my_lreg = MyLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)

# Example 0:
print(my_lreg.mse_(Y, my_lreg.predict_(X)))
# Output:
#144044.877...

# Example 1:
my_lreg.fit_(X, Y)
print(repr(my_lreg.thetas))
# Output:
#array([[367.28849...]
#		[-23.69939...]
#		[ 5.73622...]
#		[ -2.63855...]])

# Example 2:
print(my_lreg.mse_(Y, my_lreg.predict_(X)))
# Output:
#435.9325695..