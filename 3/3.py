import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# data training
X = pd.DataFrame([
    # x, y
    [5, 11],
    [13, 32],
    [2.3, 6],
    [4, 10],
    [6, 13],
    [7, 20],
    [5.7, 13],
    [12, 27]
], columns = ['x', 'y'])
y = X.pop('y')
theta0 = 100
theta1 = 2
t = np.linspace(-10, 10, 100)

y_pred = theta0 + theta1*(X)

#lr = LinearRegression()
#lr.fit(X, y)
#y_pred = lr.predict(X)
# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.title("MSE = %s"%(mean_squared_error(y, y_pred)))
plt.show()

