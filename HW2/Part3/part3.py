import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))
"""print(dataset[0])

plt.figure(figsize=(10, 7))
plt.scatter(dataset[:, 5], dataset[:, 1], s=30, alpha=0.7, edgecolor='k')
plt.title("Scatter Plot of the Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save the plot as an image file
plt.savefig("initial_data_scatter_plot.png")"""




def apply_dbscan():
    metrics = ['euclidean', 'cosine']
    eps_values = np.arange(0.1, 3, 0.3)
    min_samples_values = [2, 3, 5]

    best_config = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            for metric in metrics:

                dbscan = DBSCAN(eps=eps,
                    min_samples=min_samples,
                    metric=metric,
                    metric_params=None)
                

                predicted = dbscan.fit_predict(dataset)

                cluster_point_indices = []
                for i in range(len(predicted)):
                    if predicted[i] != -1:
                        cluster_point_indices.append(i)

                number_of_clusters = len(set(predicted)) - (1 if -1 in predicted else 0)

                if number_of_clusters >= 2:                
                    clustered_points = dataset[cluster_point_indices]
                    cluster_labels = predicted[cluster_point_indices]
                    
                    silhouette_avg = silhouette_score(clustered_points, cluster_labels)
                    best_config.append((metric, eps, min_samples, silhouette_avg, number_of_clusters))

    
    best_config = sorted(best_config, key=lambda x: x[3], reverse=True)[:4]

    for metric, eps, min_samples, _, K in best_config:

        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(dataset)
        
        silhouette_vals = silhouette_samples(dataset, labels)
        
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(silhouette_vals)), silhouette_vals)
        plt.title(f"Silhouette Plot (Metric: {metric}, Eps: {eps}, Min Samples: {min_samples})")
        plt.xlabel('Sample index')
        plt.ylabel('Silhouette value')
        plt.savefig(f"silhouette_dbscan_{metric}_eps{eps}_min{min_samples}.png")
        plt.clf()




def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def apply_hac():
    metrics = ['euclidean', 'cosine']
    linkages = ['single', 'complete']
    
    best_config = None
    best_score = -1
    config = 0

    for metric in metrics:
        for linkage in linkages:

            sil_scores = []

            hac = AgglomerativeClustering(n_clusters=None, metric=metric, linkage=linkage, distance_threshold=0)
            hac.fit(dataset)

            plt.figure(figsize=(15, 15))
            plt.title(f"HAC Dendrogram - Linkage: {linkage}, Metric: {metric}")
            plot_dendrogram(hac, truncate_mode='level', p=10)
            plt.savefig(f"dendrogram_{linkage}_{metric}.png")
            plt.clf()

            each_best_config = (linkage, metric, -1)
            each_best_score = -1

            for k in [2, 3, 4, 5]:
                hac = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage, distance_threshold=None)
                predicted = hac.fit_predict(dataset)
                silhouette_avg = silhouette_score(dataset, predicted)
                sil_scores.append(silhouette_avg)
                                
                if silhouette_avg > each_best_score:
                    each_best_score = silhouette_avg
                    each_best_config = (linkage, metric, k)

            print(f"Best K for Configuration: [{linkage}, {metric}]")
            print(f"K = {each_best_config[2]}, Silhouette Score: {each_best_score:.3f}")
            print()

            if each_best_score > best_score:
                best_score = each_best_score
                best_config = each_best_config

            plt.figure(figsize=(8, 5))
            plt.plot(range(2,6), sil_scores, marker='o')
            plt.title(f"Silhouette Scores for {linkage} Linkage with {metric} Metric")
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.savefig(f"silhouette_scores_{linkage}_{metric}.png")
            plt.clf()

    print(f"Best HAC Configuration:")
    print(f"Linkage: {best_config[0]}, Metric: {best_config[1]}, K = {best_config[2]}, Silhouette Score: {best_score:.3f}")

apply_dbscan()
apply_hac()