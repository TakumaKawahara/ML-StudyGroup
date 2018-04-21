import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD(object):

    def fit(self, X, y, lr=0.001, n_iter=20):
        self.lr = lr # Learning Rate
        self.n_iter = n_iter # Number of iteration
        self.weight = np.zeros(X.shape[1])
        self.weight = self.weight.reshape(X.shape[1],-1) # change to Number of Row to same as number of X's column.
        self.bias = np.zeros(1)
        self.cost = []
        
        for i in range(self.n_iter):
            output = np.dot(X, self.weight) + self.bias[0]
            error = (y - output)
            self.weight += self.lr * np.dot(X.T, error)
            self.bias += self.lr * error.sum()
            error_sum = (error**2).sum() / 2.0
            self.cost.append(error_sum)
        return self
            
    def predict(self, X):
        return np.dot(X, self.weight) + self.bias

def lin_regplot(X, y, model):
    matplotlib.pyplot.scatter(X, y, c='blue')
    matplotlib.pyplot.plot(X, model.predict(X), color='red')
    return None
    
        