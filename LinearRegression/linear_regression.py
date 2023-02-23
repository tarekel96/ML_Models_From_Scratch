# Author: Tarek El-Hajjaoui
# Linear Regression Model

import numpy as np

class linear_regression():
        def __init__(self, learning_rate) -> None:
                # hyperparameter that determines magnitude of weight updates
                self.learning_rate = learning_rate
                # weights - a vector with N + bias components
                self.w = None
                # feature data
                self.X = None
                # output data
                self.y = None
                # M = number of rows (data points) in X matrix, N = number of columns (features) in X matrix
                self.M, self.N = self.X.shape[0], self.X.shape[1]
                # keeps track of all the costs per weight update
                self.costs = []
        
        def fit(self, X):
                self.X = X
                # initialize weights
                self.w = np.ones((self.N + 1, 1))

                while True:
                        y_hat = self.predict(X)
                        
                        cost = self.objective_function(self.y, y_hat)
                        self.costs.append(cost)

                        gradient = self.gradient(self.y, y_hat)
                        bias_gradient = self.bias_gradient(self.y, y_hat)

                        # TODO update weights based on learning rate and gradient
                        self.w[:-1] = self.w[:-1] - self.learning_rate * gradient
                        self.w[-1] = self.w[-1] - self.learning_rate * bias_gradient

        def objective_function(self, y, y_hat):
                return (1/(2 * self.M)) * np.sum(np.square(y - y_hat))

        def gradient(self, y, y_hat):
                return (1 / self.M) * np.sum((y - y_hat) * self.X)

        def bias_gradient(self, y, y_hat):
                return (1 / self.M) * np.sum((y - y_hat))

        def predict(self, X):
                return np.dot(self.w[:-1], X) + self.w[-1] 
