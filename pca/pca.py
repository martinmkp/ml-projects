import numpy as np
from functions.calcs import Calculations

class PCA:
    """A class for executing the PCA algorithm.
    Args:
        data: An input data, either a Pandas dataframe or a NumPy array.
        dim: Dimension of the output array.
    """
    def __init__(self, data, dim = 2):
        self.data = data
        self.dim = dim

    def fit(self):
        calcs = Calculations(self.data, self.dim)
        calcs.centering()
        calcs.covar()
        calcs.eigen()
        calcs.projection()
        return calcs.proj_X
