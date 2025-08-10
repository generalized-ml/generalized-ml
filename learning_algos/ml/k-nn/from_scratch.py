import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
def plot_decision_boundary(model, X, y, resolution=200):
    """
    Plots the decision boundary of a model that has fit() and predict() methods.
    
    Parameters:
    - model: trained model with predict() method
    - X: numpy array of shape (n_samples, 2)
    - y: labels array of shape (n_samples,)
    - resolution: number of points in each grid axis
    """
    # Fit model
    # model.fit(X, y)
    
    # Create grid over feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Predict over grid
    Z = np.asarray(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    # Plot decision boundary
    plt.figure(figsize=(7,7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', edgecolor='k', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', edgecolor='k', label='Class 1')
    plt.legend()
    plt.title("Decision Boundary Visualization")
    plt.grid(True)
    plt.show()

np.random.seed(42)

# ----- Class 0 -----
# Outer horizontal blobs
blob1 = np.random.randn(30, 2) * 0.6 + np.array([-2, 0])
blob2 = np.random.randn(30, 2) * 0.6 + np.array([ 2, 0])
# Extra vertical blobs
blob3 = np.random.randn(30, 2) * 0.6 + np.array([0,  2])
blob4 = np.random.randn(30, 2) * 0.6 + np.array([0, -2])

X0 = np.vstack((blob1, blob2, blob3, blob4))
y0 = np.zeros(X0.shape[0])

# ----- Class 1 -----
# Central blob
blob5 = np.random.randn(40, 2) * 0.6 + np.array([0, 0])
# Four corner blobs
blob6 = np.random.randn(20, 2) * 0.6 + np.array([ 2,  2])
blob7 = np.random.randn(20, 2) * 0.6 + np.array([-2,  2])
blob8 = np.random.randn(20, 2) * 0.6 + np.array([ 2, -2])
blob9 = np.random.randn(20, 2) * 0.6 + np.array([-2, -2])

X1 = np.vstack((blob5, blob6, blob7, blob8, blob9))
y1 = np.ones(X1.shape[0])

# Combine
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))


class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        y = y.reshape( (X.shape[0], 1))

        self.X = np.concatenate((X, y), axis = 1)
        # print(self.X)

    def euc_distance(self, a, b):
        return sum((a-b)**2)

    def get_k_nearest(self, inp):
        dist_list = []
        for i in range(self.X.shape[0]):
            dist_list.append((i, self.euc_distance(inp,  self.X[i, :-1])))
        top_k_idx = sorted(dist_list, key = lambda x: x[1])[:self.k]
        top_k_idx = [x[0] for x in top_k_idx]
        
        class_count = Counter(self.X[top_k_idx, -1])
        return max(class_count, key = class_count.get)
    

    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            pred.append(self.get_k_nearest(X[i, : ]))

        return pred

cls = KNN(4)
cls.fit(X, y)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X, y)

plot_decision_boundary(cls,  X, y)