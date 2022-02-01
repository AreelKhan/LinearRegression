import numpy as np
from tqdm import tqdm


class LinearRegression:

    @staticmethod
    def gradient_descent(num_features, train_data, labels, learning_rate=0.05, epochs=1000):
        """
        The gradient descent algorithm. Returns the theta parameters that are iteratively modified
        to improve the accuracy of the hypothesis function.

        :param num_features: int
        :param train_data: numpy.ndarray
        :param labels: numpy.ndarray
        :param learning_rate: int
        :param epochs: int
        :return: numpy.ndarray
        """
        theta = np.ones(shape=num_features + 1)
        x_0 = np.ones_like(train_data)
        x = np.stack([x_0, train_data], axis=-1)

        for i in tqdm(range(epochs)):
            cost = np.dot(x, theta) - labels
            theta = theta - (learning_rate * (1 / len(labels)) * np.dot(x.T, cost))
        return theta
