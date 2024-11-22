import numpy as np
from Distance import Distance

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        distances = []
        
        i = 0
        for data in self.dataset:
            distance = self.similarity_function(instance, data, self.similarity_function_parameters)
            distances.append({'label': self.dataset_label[i], 'distance': distance})
            i += 1

        distances.sort(key=lambda x: x['distance'])
        nearest_neighbors = distances[:self.K]
        neighbor_labels = [neighbor['label'] for neighbor in nearest_neighbors]
        return np.bincount(neighbor_labels).argmax()

        
        


        

    

