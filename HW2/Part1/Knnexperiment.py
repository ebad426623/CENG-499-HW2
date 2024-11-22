import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import numpy as np


# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))
rsk_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)


K = [5, 10, 15, 30]
similarity_function = ["Cosine", "Minkowski", "Mahalanobis"]
similarity_function_map = {
    "Cosine": Distance.calculateCosineDistance,
    "Minkowski": Distance.calculateMinkowskiDistance,
    "Mahalanobis": Distance.calculateMahalanobisDistance
}
p_parameter = [2, 3]


combination = 1


def performKNN(k, s_func_name, s_func_para):
    accuracies = []

    s_func = similarity_function_map[s_func_name]
    for train_data_index, test_label_index in rsk_folds.split(dataset, labels):
        param = s_func_para
        

        train_data, train_labels = dataset[train_data_index], labels[train_data_index]
        test_data, test_labels =  dataset[test_label_index], labels[test_label_index]


        if(s_func == similarity_function[2]):
            param = np.linalg.inv(np.cov(train_data, rowvar = False))

        knn_model = KNN(train_data, train_labels, s_func, param, k)
        predictions = [knn_model.predict(test) for test in test_data]
        accuracies.append(np.mean(predictions == test_labels))
        

    print(np.mean(accuracies))

for k in K:
    for s_func in similarity_function:

        if s_func == similarity_function[1]:
            print(f"Hyperparamter Combination: {combination}, K = {k}, Similarity Function: {s_func}")
            combination += 1
            performKNN(k, s_func, None)

        else:
            for p in p_parameter:
                print(f"Hyperparamter Combination: {combination}, K = {k}, Similarity Function: {s_func}, p = {p}")
                para = p
                combination += 1
                performKNN(k, s_func, p)
                









