# Author: Tarek El-Hajjaoui
# Linear Regression Model

import numpy as np

class linear_regression():
        def __init__(self) -> None:
                # weights
                self.w = None
                # feature data
                self.X = None
                # output data
                self.y = None
                # M = number of rows (data points) in X matrix, N = number of columns (features) in X matrix
                self.M, self.N = self.X.shape[0], self.X.shape[1]
        
        def fit(self, X):
                self.X = X
                # weights is a vector with N + bias components
                self.w = np.ones((self.N + 1, 1))

        def predict(self):
                pass
