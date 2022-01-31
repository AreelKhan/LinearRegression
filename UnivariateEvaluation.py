import numpy as np
from sklearn.model_selection import train_test_split

from DataDefinitionFunctions import double
from LinearRegression import gradient_descent

# Data definition
X = np.random.uniform(size=400)
Y = double(X)

# Data split (hopefully the only time sklearn is used)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=69)

# Number of features
n = 1

# Parameters
theta = np.ones(shape=n+1)

parameters = gradient_descent(theta, X_train, y_train, 0.05, 1000)
print(parameters)