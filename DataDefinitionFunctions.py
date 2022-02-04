"""There is a need for labeled examples to test the linear regression implementation. The algorithm implementation
will be built up from scratch, hence, it is best to start testing it on a simple dataset. Furthermore,
poor performance can be attributed solely to errors or insufficiencies in the algorithm implementation instead of the
limited capabilities of linear regression to model a complex relation. The functions below will be used to produce
data sets with simple relations to test the linear regression implementation"""


class Functions:

    # First Relation
    @staticmethod
    def double(x):
        """
        Scales the input by 2.

        :param x: int or float or numpy.ndarray
        :return: float
        """
        return 2 * x

    # Second Relation
    @staticmethod
    def double_plus_one(x):
        """
        Scales the input by 2 and adds 1.

        :param x: int or float or numpy.ndarray
        :return: float
        """
        return 2 * x + 1

    @staticmethod
    def negate_plus_double(x_1, x_2):
        """
        Scales x_1 by -1 and adds it to 2 times x_2.

        :param x_1: int or float or numpy.ndarray
        :param x_2: int or float or numpy.ndarray
        :return: float
        """
        return -x_1 + 2 * x_2
