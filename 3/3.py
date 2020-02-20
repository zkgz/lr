import numpy as np
import matplotlib.pyplot as plt
import pandas as pda
from sklearn.linear_model import LinearRegression

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

lr = LinearRegression()s