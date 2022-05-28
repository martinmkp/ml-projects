import numpy as np
import pandas as pd

class CheckData:
    """Class that checks that the data set is usable for Sammon mapping.
    Args:
        dataframe: Pandas or NumPy dataframe.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def check_dataframe_type(self):
        """Checks dataframe type. Pandas dataframes are changed to NumPy arrays.
        """
        self.dataframe
        err_message = 'wrong type: Type not in (np.ndarray, pd.Series)'
        if isinstance(self.dataframe, (np.ndarray, pd.Series)):
            if not isinstance(self.dataframe, (np.ndarray)):
                self.dataframe = self.dataframe.to_numpy()
            return 1
        else:
            print(data_type)
            raise ValueError(err_message)
            return 0

    def check_nan_values(self):
        """Checks numpy array for nan-values.
        """
        err_message = 'wrong type: Array contains NaN-values'
        nan_total = np.count_nonzero(np.isnan(self.dataframe))
        if np.isnan(nan_total) == 0:
            return 1
        else:
            raise ValueError(err_message)
            return 0
