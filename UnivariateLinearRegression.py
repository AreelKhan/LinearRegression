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
    This is the hypothesis function. The parameters are model weights, theta, and the inp is the training example.
    :param parameters: numpy.ndarray
    :param inp: numpy.ndarray
    :return:
    """
    return np.dot(parameters, inp)


# Value of derivative of cost function
costs = [h(theta, X_train.iloc[i]) - y_train.iloc[i] for i in range(len(X_train))]
average_cost = 1 / m * np.array(costs).sum()

print(X_train.iloc[0])


# TODO: Figure out the gradient descent for loop portion
# TODO: add docs for avg_cost function
def avg_cost(parameters, train, test):
    output = np.zeros_like(parameters)
    print(type(output))
    for i in range(len(X_train)):
        ((h(parameters, train.iloc[i]) - test.iloc[i]) * train.iloc[i])
    return output


print(avg_cost(theta, X_train, X_test))

# Learning rate - alpha
alpha = 1

# Gradient Descent
theta = theta - alpha * average_cost * X_train

print(original_parameters)
print(theta)
