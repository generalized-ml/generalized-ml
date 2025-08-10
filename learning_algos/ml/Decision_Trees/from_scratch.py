import numpy as np
import matplotlib.pyplot as plt

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

# # Plot
# plt.figure(figsize=(7,7))
# plt.scatter(X[y==0][:,0], X[y==0][:,1], c='blue', label='Class 0', edgecolor='k')
# plt.scatter(X[y==1][:,0], X[y==1][:,1], c='red', label='Class 1', edgecolor='k')
# plt.legend()
# plt.title("Complex Non-Linear 2D Data with Y-axis Clusters")
# plt.grid(True)
# plt.show()



class MyDT():
    def __init__(self, max_depth, min_sample):
        self. max_depth = max_depth
        self. min_sample = min_sample
        self.tree = {} # final aim is to build this tree diction {feature: 1, value : m left :{<same>} , right:}
        self.input_size = None
        self.feature_size = None


    def gini_index(self, groups):

        gini = []

        for gr in groups:
            gr_size = gr.shape[0]
            sum_  = 0
            if gr_size>0:
                for t in list(set(gr[:, -1])):
                    p = sum(gr[:, -1]==t)/gr_size
                    sum_ = sum_+ (p**2)
            gini.append((1- sum_)*(gr_size))

        return sum(gini)
    



    def get_split(self, data, feat, value):
        left = []
        right = []
        
        for x in data:
            if x[feat]>=value:
                right.append(x)
            else:
                left.append(x)
        return np.asarray(left), np.asarray(right)



        
    def split_data(self, x_train):
        b_score = 999999
        return_dict = {}
        if x_train.shape[0]==0:
            return -1
        

        dist_class = list(set(x_train[:, -1]))
        if len(dist_class)==1:
            return dist_class[0]
        
        if x_train.shape[0]<=self.min_sample:
            clss_dst = 0
            best_cls = 0
            for clas in dist_class:
                num = sum(x_train[:, -1]==clas)
                if num >clss_dst:
                    clss_dst = num
                    best_cls = clas
            return best_cls
            

        for feat in range(self.feature_size):
            for value in x_train[:, feat]:
                group = self.get_split(x_train, feat, value)
                gini = self.gini_index(group)
                if gini< b_score:
                    b_score = gini
                    left = group[0] 
                    if left.shape[0]==0:
                        left = -1
                    right = group[1] 
                    if right.shape[0]==0:
                        right = -1

                    return_dict = {"feature" : feat, "value" :value, 
                                   "left": left, "right": right}
                    

        return return_dict


    def build_recurse(self, tree):

        try:
            int(tree)
        except:
            tree["left"] = self.split_data(tree["left"])
            tree["left"] = self.build_recurse(tree["left"])
            
            tree["right"] = self.split_data(tree["right"])
            tree["right"] = self.build_recurse(tree["right"])
        
        return tree



    def fit(self, X, y):
        self.input_size = X.shape[0]
        self.feature_size =  X.shape[1]
        y = y.reshape(self.input_size, 1)

        X = np.concatenate((X, y), axis = 1)
        tree  = self.split_data(X)
        self.tree = self.build_recurse(tree)
        
    def pred_recursive(self, dict_, inp):
        try:
            int(dict_)
            return dict_
        except:
            if inp[dict_["feature"]]>=dict_["value"]:
                return self.pred_recursive(dict_["right"], inp)
            else:
                return self.pred_recursive(dict_["left"], inp)
        
        return None
    
    def predict(self, inp):
        pred = []
        for x in range(inp.shape[0]):
            pred.append(self.pred_recursive(self.tree, inp[x, :]))
        return pred


   

cls = MyDT(5, 10)
cls.fit(X, y)


# print(cls.tree)


plot_decision_boundary(cls,  X, y)




