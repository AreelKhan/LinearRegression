"""There is a need for labeled examples to test the linear regression implementation. Ideally, the relation in the
data is very simple. The algorithm implementation will be built up from scratch, hence, it is best to start testing
it on a simple dataset. Furthermore, poor performance can be attributed solely to errors or insufficiencies in the
algorithm implementation instead of the limited abilities of linear regression to model a complex relation."""


# First Relation
def func_one(input):
    """
    This is a very simple linear mapping. Scales the input by 2.

    :param input: int or float or numpy.ndarray.
    :return: float
    """
    return 2 * input


# Second Relation
def func_two(input):
    """
    This is a very simple linear mapping. Scales the input by -1 and adds 3.

    :param input: int or float or numpy.ndarray.
    :return: float
    """
    return -1 * input + 3


