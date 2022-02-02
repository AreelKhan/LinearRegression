import numpy as np
from tqdm import tqdm


class LinearRegression:

    @staticmethod
    def gradient_descent(train_data, labels, learning_rate=0.05, epochs=1000):
        """
        The gradient descent algorithm. Returns the theta parameters that are iteratively modified
        to improve the accuracy of the hypothesis function.

        :param train_data: numpy.ndarray
        :param labels: numpy.ndarray
        :param learning_rate: int
        :param epochs: int
        :return: numpy.ndarray
        """
        if len(train_data.shape) == 1:
            num_features = 1
        else:
            num_features = train_data.shape[1]

        theta = np.ones(shape=num_features + 1)
        x_0 = np.ones_like(labels)
        x = np.column_stack([x_0, train_data])

        for _ in tqdm(range(epochs)):
            cost = np.dot(x, theta) - labels
            theta = theta - (learning_rate * (1 / len(labels)) * np.dot(x.T, cost))
        return theta
