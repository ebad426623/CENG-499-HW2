import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import numpy as np


# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))
rsk_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)


K = [3, 5, 10, 15, 30]

similarity_function = ["Cosine", "Minkowski", "Mahalanobis"]
similarity_function_map = {
    "Cosine": Distance.calculateCosineDistance,
    "Minkowski": Distance.calculateMinkowskiDistance,
    "Mahalanobis": Distance.calculateMahalanobisDistance
}

p_parameter = [2, 3, 4]
combination = 1

best_combination = [0, "", None]
best_mean = 0
best_interval = 0
best_confidence_interval = [0, 0]

def performKNN(k, s_func_name, s_func_para):
    global best_mean, best_combination, best_interval, best_confidence_interval
    accuracies = []

    s_func = similarity_function_map[s_func_name]
    for train_data_index, test_label_index in rsk_folds.split(dataset, labels):
        param = s_func_para
        

        train_data, train_labels = dataset[train_data_index], labels[train_data_index]
        test_data, test_labels =  dataset[test_label_index], labels[test_label_index]


        if(s_func_name == similarity_function[2]):
            param = np.linalg.inv(np.cov(m = train_data, rowvar = False))

        knn_model = KNN(train_data, train_labels, s_func, param, k)
        predictions = [knn_model.predict(test) for test in test_data]
        accuracies.append(np.mean(predictions == test_labels))
        

    mean_accuracy = np.mean(accuracies)
    interval = 1.96 * (np.std(accuracies)/np.sqrt(len(accuracies)))
    confidence_interval = [mean_accuracy - interval, mean_accuracy + interval]

    print(f"Mean Accuracy: {mean_accuracy:.2f}, Interval: {interval:.2f}, Confidence Interval ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")

    if mean_accuracy > best_mean:
        best_mean = mean_accuracy
        best_interval =  interval
        best_confidence_interval = confidence_interval
        best_combination = [k, s_func_name, s_func_para]


for k in K:
    for s_func in similarity_function:

        if s_func != similarity_function[1]:
            print(f"Hyperparamter Combination: {combination}, K = {k}, Similarity Function: {s_func}")
            combination += 1
            performKNN(k, s_func, None)
            print()

        else:
            for p in p_parameter:
                print(f"Hyperparamter Combination: {combination}, K = {k}, Similarity Function: {s_func}, p = {p}")
                para = p
                combination += 1
                performKNN(k, s_func, p)
                print()

if best_combination[1] == similarity_function[1]:
    print(f"Best Hyperparameters: K = {best_combination[0]}, "
          f"Similarity Function: {best_combination[1]}, "
          f"p = {best_combination[2]:.2f}, "
          f"Accuracy = {best_mean:.2f}, "
          f"Interval = {best_interval:.2f}, "
          f"Confidence Interval ({best_confidence_interval[0]:.2f}, {best_confidence_interval[1]:.2f})")

else:
    print(f"Best Hyperparameters: K = {best_combination[0]}, "
          f"Similarity Function: {best_combination[1]}, "
          f"Accuracy = {best_mean:.2f}, "
          f"Interval = {best_interval:.2f}, "
          f"Confidence Interval ({best_confidence_interval[0]:.2f}, {best_confidence_interval[1]:.2f})")
