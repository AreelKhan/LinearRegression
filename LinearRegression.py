import numpy as np


def gradient_descent(theta, examples, labels, learning_rate, epochs):
    """
    The gradient descent algorithm. Returns the theta parameters that are iteratively modified to improve the accuracy
    of the hypothesis function.

    :param theta: numpy.ndarray
    :param examples: numpy.ndarray
    :param labels: numpy.ndarray
    :param learning_rate: int
    :param epochs: int
    :return: numpy.ndarray
    """
    x_0 = np.ones_like(examples)
    x = np.stack([x_0, examples], axis=-1)

    for i in range(epochs):
        cost = np.dot(x, theta) - labels
        theta = theta - (learning_rate * (1 / len(labels)) * np.dot(x.T, cost))
    return theta
