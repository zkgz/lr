import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
image = [
    [100, 100, 100, 0, 0, 0], 
    [100, 100, 100, 0, 0, 0], 
    [100, 100, 100, 0, 0, 0], 
    [100, 100, 100, 0, 0, 0], 
    [100, 100, 100, 0, 0, 0], 
    [100, 100, 100, 0, 0, 0], 
]

filter = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
]
plt.imshow(image, cmap="gray")
plt.show()