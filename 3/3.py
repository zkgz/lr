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


X = pd.DataFrame([
    # x, y
    [5, 11],
    [2.3, 6],
    [4, 10]
], columns = ['x', 'y'])
y = X.pop('y')


theta0 = 10
theta1 = 1
y_pred = theta0 + theta1*(X)

t = np.linspace(-10, 10, 100)
f = np.square(t)
#g = np.cos(t)
plt.figure(2)
plt.plot(t, f)
#plt.plot(t, g)


lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

# Plot outputs
plt.figure(0)
plt.scatter(X, y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
#plt.gca().set_xlim([0,None])
plt.title("theta0 = %s\n theta1 = %s\nMSE = %s"%(lr.coef_, lr.coef_, mean_squared_error(y, y_pred)))


plt.figure(1)
X = [0, 1, 2, 3, 4, 5, 10]
y = [28, 19, 11, 6.16, 2.69, 1.2, 23]
plt.scatter(X, y,  color='black')
plt.plot()
plt.show()