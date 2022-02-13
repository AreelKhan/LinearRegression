import numpy as np
from tqdm import tqdm


class LinearRegression:

    def __init__(self):
        self.parameters = None
        self.num_features = None

    def fit(self, X, y, learning_rate=0.1, epochs=3000):
        """
        Fit the linear regression model using the gradient descent algorithm. The theta parameters are
        iteratively modified to improve the accuracy of the hypothesis function. The trained parameters are stored
        in the LinearRegression instance.

        :param X: numpy.ndarray
            The features of training data
        :param y: numpy.ndarray
            The labels of the training data
        :param learning_rate: int
            The learning rate
        :param epochs: int
            The number of epochs/iterations of gradient descent
        :return: None
        """
        if len(X.shape) == 1:
            self.num_features = 1
        else:
            self.num_features = X.shape[1]

        theta = np.ones(shape=self.num_features + 1)
        x_0 = np.ones_like(y)
        x = np.column_stack([x_0, X])
        for _ in tqdm(range(epochs)):
            cost = np.dot(x, theta) - y
            theta = theta - (learning_rate * (1 / len(y)) * np.dot(x.T, cost))
        self.parameters = theta

    def pred(self, X):
        """
        Make predictions using the trained linear regression model.

        Constraints:
        fit() method must be called first

        :param X: numpy.ndarray
            The data on which to make predictions
        :return: y_pred
            The predictions on the data
        """
        if type(self.parameters) is None:
            print("fit() method must be called first")
            return None
        if self.num_features == 1:
            X = X.reshape((len(X), 1))
        y_pred = []
        for i in range(len(X)):
            pred = np.dot(self.parameters, np.insert(X[i], 0, 1))
            y_pred.append(pred)
        return y_pred

    def get_params(self):
        """
        Get the trained parameters

        :return: numpy.ndarray
            The trained parameters
        """
        return self.parameters
