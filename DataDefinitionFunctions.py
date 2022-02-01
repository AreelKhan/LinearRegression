"""There is a need for labeled examples to test the linear regression implementation. The algorithm implementation
will be built up from scratch, hence, it is best to start testing it on a simple dataset. Furthermore,
poor performance can be attributed solely to errors or insufficiencies in the algorithm implementation instead of the
limited capabilities of linear regression to model a complex relation. The functions below will be used to produce
data sets with simple relations to test the linear regression implementation"""


# First Relation
def double(x):
    """
    This is a very simple linear mapping. Scales the input by 2.

    :param x: int or float or numpy.ndarray.
    :return: float
    """
    return 2 * x


# Second Relation
def double_plus_one(x):
    """
    This is a very simple linear mapping. Scales the input by 2 and adds 1.

    :param x: int or float or numpy.ndarray.
    :return: float
    """
    return 2 * x + 1


