import numpy as np
from sklearn.decomposition import PCA

class Calculations:
    """Class for calculating Sammon mapping.
    args:
        array: Numpy array
        pca_dim: PCA dimension after reduction.
    """
    def __init__(self, array, pca_dim):
        self.X = array
        self.pca = PCA(n_components = pca_dim)
        self.Y = self.pca.fit_transform(self.X)
        self.D_X = np.zeros((self.X.shape[0], self.X.shape[0]))
        self.D_Y = np.zeros((self.Y.shape[0], self.Y.shape[0]))
        self.const = 0

    def euclidean_dist(self, mat):
        """Calculates distance matrix.
        Args:
            mat: A matrix of real numbers.
        """
        dist = np.zeros((mat.shape[0], mat.shape[0]))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                dist[i, j] = np.linalg.norm(mat[i,:] - mat[j,:])
        return dist

    def initialize_distances(self):
        """Initializes distance matrices for X and Y.
        """
        self.D_X = self.euclidean_dist(self.X)
        self.D_Y = self.euclidean_dist(self.Y)

    def constant(self):
        """Calculates a constant from the distance matrix D_X.
        """
        for i in range(self.D_X.shape[0]):
            for j in range(i+1, self.D_X.shape[1]):
                    self.const += self.D_X[i, j]

    def first_derivative(self, i, k):
        """First derivative to determine new values for D_Y.
        Args:
            i: Row index for Y
            k: Column index for Y
        """
        total_d = 0
        step1 = 0
        step2 = 0
        for j in range(self.Y.shape[0]):
            if j != i:
                step1 = -2 * (self.D_X[j, i] - self.D_Y[j, i]) / self.D_X[j, i]
                step2 = (self.Y[i, k] - self.Y[j, k]) /  self.D_Y[j, i]
                total_d += step1 * step2
        result = np.divide(total_d, self.const)

        return result

    def second_derivative(self, i, k):
        """Second derivative to determine new values for D_Y.
        Args:
            i: Row index for Y
            k: Column index for Y
        """
        total_d = 0
        step1 = 0
        step2 = 0
        step3 = 0
        step4 = 0
        for j in range(self.Y.shape[0]):
            if j != i:
                step1 = 1 / (self.D_X[j, i] * self.D_Y[j, i])
                step2 = (self.D_X[i, j] - self.D_Y[i, j])
                step3 = np.square(self.Y[i, k] - self.Y[j, k]) / self.D_Y[i, j]
                step4 = 1 + (self.D_X[i, j] - self.D_Y[i, j]) / self.D_Y[i, j]
                total_d += step1 * (step2 - step3 * step4)
        result = -2 * np.divide(total_d, self.const)

        return result

    def execute_iteration(self, rate, iters):
        for r in range(iters):
            for i in range(self.Y.shape[0]):
                for k in range(self.Y.shape[1]):
                    nominator =  self.first_derivative(i, k)
                    denominator = np.absolute(self.second_derivative(i, k))
                    self.Y[i, k] -= rate * np.divide(nominator, denominator)
            self.D_Y = self.euclidean_dist(self.Y)
