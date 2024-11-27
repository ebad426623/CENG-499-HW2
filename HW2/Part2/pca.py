import numpy as np
# https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e
class PCA:
    def __init__(self, projection_dim: int):
        """
        Initializes the PCA method
        :param projection_dim: the projection space dimensionality
        """
        self.projection_dim = projection_dim
        # keeps the projection matrix information
        self.projection_matrix = None

    def fit(self, x: np.ndarray) -> None:
        """
        Applies the PCA method and obtains the projection matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should assign the resulting projection matrix to self.projection_matrix
        """

        # Standardizing my Data along the features
        standardized_x = (x - x.mean(axis = 0)) / x.std(axis=0)

        # Finding Covaraince Matrix
        covariance_matrix = np.cov(standardized_x, rowvar=False)

        # Eigendecomposition
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

        # Sorting in decending order to find order of importance
        order_of_importance = np.argsort(eigen_values)[::-1]

        # Calculating and assigning projection matrix
        sorted_eigenvectors = eigen_vectors[:, order_of_importance]
        self.projection_matrix = sorted_eigenvectors[:, :self.projection_dim]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        After learning the projection matrix on a given dataset,
        this function uses the learned projection matrix to project new data instances
        :param x: data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        this function should utilize self.projection_matrix for the operations
        """

        standardized_x = (x - x.mean(axis = 0)) / x.std(axis=0)
        return np.matmul(standardized_x, self.projection_matrix)