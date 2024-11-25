import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

max_K = 10
num_runs = 10
num_repeats = 10

def find_elbow_and_plot(dataset, dataset_name):
    print(dataset_name)

    avg_loss = []
    loss_conf_intervals = []
    avg_silhouette = []
    silhouette_conf_intervals = []

    for k in range(2, max_K + 1):
        # Collect loss and silhouette scores for each run
        losses = []
        silhouettes = []

        for _ in range(num_runs):
            # Run k-medoids with repeats
            repeat_losses = []
            repeat_silhouettes = []

            for _ in range(num_repeats):
                kmedoids = KMedoids(n_clusters=k, random_state=None, method="pam")
                kmedoids.fit(dataset)
                inertia = kmedoids.inertia_
                repeat_losses.append(inertia)

                silhouette = silhouette_score(dataset, kmedoids.labels_)
                repeat_silhouettes.append(silhouette)

            losses.append(np.mean(repeat_losses))
            silhouettes.append(np.mean(repeat_silhouettes))

        # Compute the average and confidence intervals for loss and silhouette
        avg_loss_val = np.mean(losses)
        loss_conf_interval = 1.96 * (np.std(losses) / np.sqrt(len(losses)))
        avg_loss.append(avg_loss_val)
        loss_conf_intervals.append(loss_conf_interval)

        avg_silhouette_val = np.mean(silhouettes)
        silhouette_conf_interval = 1.96 * (np.std(silhouettes) / np.sqrt(len(silhouettes)))
        avg_silhouette.append(avg_silhouette_val)
        silhouette_conf_intervals.append(silhouette_conf_interval)

        # Print the results for this k
        print(f"For K = {k}")
        print(f"Mean Loss: {avg_loss_val:.2f}, Interval: {loss_conf_interval:.3f}")
        print(f"Mean Silhouette: {avg_silhouette_val:.2f}, Interval: {silhouette_conf_interval:.3f}")
        print()

    ks = range(2, max_K + 1)

    # Plot Loss
    plt.figure()
    plt.errorbar(ks, avg_loss, yerr=loss_conf_intervals, capsize=5, label="Loss", fmt='-o')
    plt.title(f"Elbow Method (Loss) for {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(f"{dataset_name}_kmedoids_loss_plot.png")

    # Plot Silhouette
    plt.figure()
    plt.errorbar(ks, avg_silhouette, yerr=silhouette_conf_intervals, capsize=5, label="Silhouette", fmt='-o')
    plt.title(f"Elbow Method (Silhouette) for {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.legend()
    plt.savefig(f"{dataset_name}_kmedoids_silhouette_plot.png")

# Call the function for both datasets
find_elbow_and_plot(dataset1, "Dataset 1")
find_elbow_and_plot(dataset2, "Dataset 2")
