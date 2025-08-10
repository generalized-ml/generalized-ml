import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate points for class 0
x0 = np.random.randn(50, 2) - np.array([2, 2])  # shifted to bottom-left
y0 = np.zeros(50)

# Generate points for class 1
x1 = np.random.randn(50, 2) + np.array([2, 2])  # shifted to top-right
y1 = np.ones(50)

# Combine
X = np.vstack((x0, x1))
y = np.hstack((y0, y1))

X = np.vstack((X, [[-1, 1.558],[0.8, 2.2], [-3, -0.6] ]))
y = np.hstack((y, [0, 0, 1]))

#we have to implement the decision boundry usng the data two dimentional data is this 
class LrClassifier():
    def __init__(self, input_x, epochs= 200, learning_rate = 0.01):
        self.input_size = input_x.shape[1]
        self.weigths = np.random.uniform(-1, 1, size=self.input_size + 1)
        self.epochs = epochs
        self.learning_rate = learning_rate

        #min max scalar TO DO
        self.x_min = np.min(input_x, axis = 1)
        self.x_max = np.max(input_x, axis = 1)
        self.x = input_x
        self.all_loss = []
    def calculate_loss(self, pred, y):
        return np.sum((pred-y)**2)

    def normalize_input(self, x_vec):
        x_vec
        return None
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def predict_train(self, inp):
        return self.sigmoid(inp @ self.weigths)
    
    def calculate_grd(self, X, pred, y):
        return X.T @ (pred - y)

    def fit(self , X,y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        #how we wil train  batch or one at a time
        print(X.shape)
        
        for e in range(self.epochs):
            pred = self.predict_train(X)
            self.all_loss.append(self.calculate_loss(pred, y))
            gradient = self.calculate_grd(X, pred , y)
            self.weigths  = self.weigths  - self.learning_rate * gradient

    def predict(self, inp):
        inp = np.hstack((inp, np.ones((inp.shape[0], 1))))
        return self.sigmoid(inp @ self.weigths)
    


clf = LrClassifier(X)

clf.fit(X, y)


# Step 3: Create grid and get predictions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = clf.predict(grid).reshape(xx.shape)

# Step 4: Plot decision regions + boundary + data
plt.figure(figsize=(7, 7))
# Color regions based on predicted class
plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.bwr)
# Decision boundary (change points between classes)
plt.contour(xx, yy, preds, levels=[0.5], linewidths=2, colors='green')
# Data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', edgecolor='k', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', edgecolor='k', label='Class 1')
plt.legend()
plt.title("Decision Boundary using Predict()")

loss_history = clf.all_loss
# Plot loss curve
plt.figure(figsize=(6,4))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.xlabel("Iteration / Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.grid(True)
plt.show()

plt.show()


