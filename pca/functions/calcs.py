import numpy as np

class Calculations:
    """Class for performing pca.
    args:
        X: A numpy array.
        dim: PCA dimensions after reduction.
        """
    def __init__(self, X, dim):
        self.X = X
        self.dim = dim
        self.X_mean = None
        self.cen_X = None
        self.cov = None
        self.eigen_val = None
        self.eigen_vec = None
        self.proj_X = None

    def centering(self):
        self.X_mean = np.mean(self.X, axis = 0)
        self.cen_X = self.X - self.X_mean

    def covar(self):
        self.cov = np.cov(self.cen_X.T, bias = True)

    def eigen(self):
        self.eigen_val, self.eigen_vec = np.linalg.eig(self.cov)
        order = self.eigen_val.argsort()[::-1]
        self.eigen_val = self.eigen_val[order]
        self.eigen_vec = self.eigen_vec[:, order]

    def projection(self):
        self.proj_X = self.cen_X @ self.eigen_vec[:, :self.dim]
