import numpy as np
from tqdm import tqdm


class LinearRegression:

    def __init__(self):
        self.parameters = None

    def fit(self, X, y, learning_rate=0.05, epochs=1000):
        """
        The gradient descent algorithm. Returns the theta parameters that are iteratively modified
        to improve the accuracy of the hypothesis function.

        :param X: numpy.ndarray
        :param y: numpy.ndarray
        :param learning_rate: int
        :param epochs: int
        :return: numpy.ndarray
        """
        if len(X.shape) == 1:
            num_features = 1
        else:
            num_features = X.shape[1]

        theta = np.ones(shape=num_features + 1)
        x_0 = np.ones_like(y)
        x = np.column_stack([x_0, X])
        for _ in tqdm(range(epochs)):
            cost = np.dot(x, theta) - y
            theta = theta - (learning_rate * (1 / len(y)) * np.dot(x.T, cost))
        self.parameters = theta

    def pred(self, X):
        y_pred = []
        for i in range(len(X)):
            pred = np.dot(self.parameters, np.insert(X[i], 0, 1))
            y_pred.append(pred)
        return y_pred
