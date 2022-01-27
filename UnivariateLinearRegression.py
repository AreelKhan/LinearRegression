from sklearn.model_selection import train_test_split
from DataSetProduction import func_one
import numpy as np

# Data definition
x = np.arange(-100, 100.5, step=0.5)
y = func_one(x)

# Data split (hopefully the only time I use sklearn)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=69)

# Number of training examples
m = len(X_train)



