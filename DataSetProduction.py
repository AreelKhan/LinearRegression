"""There is a need for labeled examples to test the linear regression implementation. Ideally, the relation in the
data is very simple. The algorithm implementation will be built up from scratch, hence, it is best to start testing
it on a simple dataset. Furthermore, poor performance can be attributed solely to errors or insufficiencies in the
algorithm implementation instead of the limited abilities of linear regression to model a complex relation."""


# First Relation
def func_one(x):
    """
    This is a very simple linear mapping. Scales the input by 2.

    :param x: int or float or numpy.ndarray.
    :return: float
    """
    return 2 * x


# Second Relation
def func_two(x):
    """
    This is a very simple linear mapping. Scales the input by -1 and adds 3.

    :param x: int or float or numpy.ndarray.
    :return: float
    """
    return -1 * x + 3


