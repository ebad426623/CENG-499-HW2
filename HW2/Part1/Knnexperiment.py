import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold


# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=42)
rsk_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)



K = [5, 10, 15, 30]
similarity_function = ["Cosine", "Minkowski", "Mahalanobis"]

p_parameter = [2, 3]

combination = 1

for k in K:
    for s_func in similarity_function:
        if s_func != similarity_function[1]:
            print(f"Hyperparamter Combination: {combination}, K = {k}, Similarity Function: {s_func}")
            combination += 1
        else:
            for p in p_parameter:
                print(f"Hyperparamter Combination: {combination}, K = {k}, Similarity Function: {s_func}, p = {p}")
                combination += 1









