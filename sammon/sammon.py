import numpy as np
from functions.check_data import CheckData
from functions.calcs import Calculations
from sklearn.metrics import euclidean_distances

class Sammon:
    """A class for executing the Sammon mapping algorithm.
    Args:
        data: An input data, either a Pandas dataframe of NumPy array.
        dim: Dimension of the output array.
        rate: Rate of learning, 0.3 by default.
        iters: Number of iterations, 100 by default.
    """
    def __init__(self, data, dim = 2, rate = 0.3, iters = 100):
        self.data = data
        self.dim = dim
        self.rate = rate
        self.iters = iters

    def fit(self):
        # Checks the data
        check = CheckData(self.data)
        check_type = check.check_dataframe_type()
        if check_type == 0:
            return
        check_nan = check.check_nan_values()
        if check_nan == 0:
            return
        calcs = Calculations(self.data, self.dim)
        calcs.constant()
        calcs.execute_iteration(self.rate, self.iters)

        return calcs.Y
