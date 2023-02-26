# Author: Tarek El-Hajjaoui
# Linear Regression Model

import numpy as np

class LinearRegression():
        def __init__(self, lr=0.01, epochs=1000) -> None:
                # hyperparameter lr = learning rate which determines magnitude of weight updates
                self.lr = lr
                # number of iterations
                self.epochs = epochs
                # weights - a vector with N + bias components
                self.w = None
                # feature data
                self.X = None
                # output data
                self.y = None
                # keeps track of all the costs per weight update
                self.costs = []
        
        def objective_function(self, y_residual):
                return (1/(2 * self.M)) * np.dot(y_residual.T, y_residual)

        def gradient(self, y_residual):
                return (1 / self.M) * np.dot(self.X.T, y_residual)

        def bias_gradient(self, y_residual):
                return (1 / self.M) * np.sum(y_residual)

        def predict(self, X):
                return np.dot(self.w[:-1], self.X.T) + self.w[-1]

        def fit(self, X, y):
                self.X = X
                self.y = y
                # M = number of rows (data points) in X matrix, N = number of columns (features) in X matrix
                self.M, self.N = self.X.shape[0], 1
                if len(self.X.shape) > 1:
                        self.N = self.X.shape[1]

                # initialize weights
                self.w = np.ones((self.N + 1, ))

                for _ in range(self.epochs):
                        # linear prediction
                        y_hat = self.predict(X)
                        
                        # residual
                        error = y_hat - self.y

                        # keep track of costs
                        cost = self.objective_function(y_residual=error)
                        self.costs.append(cost)
                        
                        # calculate gradient vectors for weight updates
                        gradient = self.gradient(y_residual=error)
                        bias_gradient = self.bias_gradient(y_residual=error)

                        # update weights based on learning rate and gradient
                        self.w[:-1] = self.w[:-1] - self.lr * gradient
                        self.w[-1] = self.w[-1] - self.lr * bias_gradient
