# Make A Simple Dataset
# We can start using the PCA by creating our own little dataset.
#  For this lesson, we'll make a 3D (three-dimensional) dataset of 200 points:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
# Creating 200-point 3D dataset
X = np.dot(np.random.random(size=(3, 3)), np.random.normal(size=(3, 200))).T
# Plotting the dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
plt.title("Scatter Plot of Original Dataset")
plt.show()