# creating the classifier of a moon dataset using NN from scratch 



from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
def vec_dot(x1, x2):
    out = 0
    for p, q in zip(x1, x2):
        out = out  +p*q
    return out

#nn will have two sections feedforward and backpropogation
import random
class NN():
    def __init__(self, input_shape, learning_rate):
        self.leraning_rate = learning_rate
        self.input_shape = input_shape
        self.weights_1  = random.sample(range(0,input_shape*10), input_shape)
        self.weights_1 = [x /input_shape for x in self.weights_1 ]
        self.bais = random.sample(self.weights_1, 1)


    def feed_forward(self, input):

        assert len(input)== self.input_shape , 'shape not matching'
        logit  = vec_dot(input, self.weights_1 ) +  self.bais
        if logit>0:
            return 1
        else:
            return 0


    def train(self, X_train, y, epoch):
        for e in range(epoch):
            for train, y_ in zip(X_train, y):
                pred = self.feed_forward(train)

                for i in range(len(self.weights_1)):
                    self.weights_1[i] -= self.leraning_rate*(pred - y_)*train[i]
                
                self.bais -= self.leraning_rate*(pred - y_)*1
    
model = NN(2, learning_rate = 0.1)



def accuracy(pred, true):
    return sum([1*(x==y) for x, y in zip(pred, true)])/len(true)

X_test, y_test = make_moons(n_samples=20, noise=0.2, random_state=42)
pred = []
for i in range(20):
    pred.append(model.feed_forward(X_test[i, :]))

print(accuracy(pred, y_test))

print(model.weights_1, model.bais)   
print("------------traiing-----------")
model.train(X, y, 300)
print(model.weights_1, model.bais)   

pred = []
for i in range(20):
    pred.append(model.feed_forward(X_test[i, :]))

print(accuracy(pred, y_test))

