import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DataSetProduction import func_one

# Data definition
x_1 = np.arange(-100, 100.5, step=0.5)
x_0 = np.ones_like(x_1)
y = func_one(x_1)
data = pd.DataFrame({'x_0': x_0, 'x_1': x_1, "y": y})

# Data split (hopefully the only time sklearn is used)
X_train, X_test, y_train, y_test = train_test_split(data[["x_0", "x_1"]], data["y"], test_size=0.30, random_state=69)

# Number of training examples
m = len(X_train)

# Number of variables (1 variable because univariate linear regression)
n = X_train.shape[1] - 1

# Parameters
original_parameters = np.random.randint(low=-10, high=10, size=n + 1)
theta = original_parameters


# Hypothesis Function
def h(parameters, inp):
    """
    This is the hypothesis function.
    parameters are the model weights, theta.
    inp is the i-th training example.

    :param parameters: numpy.ndarray
    :param inp: numpy.ndarray
    :return: float
    """
    return np.dot(parameters, inp)


def gradient_descent(parameters, examples, labels, hypothesis, learning_rate, epochs):
    summation = 0
    for i in range(len(examples)):
        summation += hypothesis(parameters, examples.iloc[i]) - labels.iloc[i]
    parameters = parameters - learning_rate/len(examples) * summation * examples.iloc[]

# Value of derivative of cost function
costs = [X_train.iloc[i] * (h(theta, X_train.iloc[i]) - y_train.iloc[i]) for i in range(len(X_train))]

# TODO: Figure out the gradient descent for loop portion
# TODO: add docs for avg_cost function
