import numpy as np
import math

class Distance:

    @staticmethod
    def calculateCosineDistance(x, y):
        dot_prodcut = np.dot(x, y)
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        cosine_distance = 1
        if x_norm != 0 and y_norm != 0:
            cosine_similarity = (dot_prodcut)/(x_norm * y_norm)
            cosine_distance = 1 - cosine_similarity    
        return cosine_distance

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        return np.sum(np.abs(x - y) ** p) ** (1 / p)
    
    @staticmethod
    def calculateMahalanobisDistance(x,y, S_minus_1):
        delta = x - y
        return np.sqrt(np.dot(np.dot(delta.T, S_minus_1), delta))