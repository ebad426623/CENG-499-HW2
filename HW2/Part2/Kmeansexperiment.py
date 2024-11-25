import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))



max_K = 10
num_runs = 10



def find_elbow_and_plot(dataset, dataset_name):
    print(dataset_name)

    avg_loss = []
    loss_interval = 0
    l_i = []
    loss_conf_intervals = []

    avg_silohoutte = []
    silohoutte_interval = 0
    s_i = []
    silohoutte_conf_intervals = []
    

    for k in range(2, max_K + 1):
        losses = []
        silohouttes = []

        for _ in range(num_runs + 1):
            
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=None, n_init=10)
            kmeans.fit(dataset)
            loss = kmeans.inertia_
            losses.append(loss)

            # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.silhouette_score.html
            silohoutte = silhouette_score(dataset, kmeans.fit_predict(dataset))
            silohouttes.append(silohoutte)
        

        avg_l = np.mean(losses)
        avg_loss.append(avg_l)
        loss_interval = 1.96 * (np.std(losses)/np.sqrt(len(losses)))
        l_i.append(loss_interval)
        l_cf = [avg_l - loss_interval, avg_l + loss_interval]
        loss_conf_intervals.append(l_cf)


        
        avg_s = np.mean(silohouttes)
        avg_silohoutte.append(avg_s)
        silohoutte_interval = 1.96 * (np.std(silohouttes)/np.sqrt(len(silohouttes)))
        s_i.append(silohoutte_interval)
        s_cf = [avg_s - silohoutte_interval, avg_s + silohoutte_interval]
        silohoutte_conf_intervals.append(s_cf)

        print(f"For K = {k}")
        print(f"Mean Loss: {avg_l:.2f}, Interval: {loss_interval:.3f}, Confidence Interval ({l_cf[0]:.3f}, {l_cf[1]:.3f})")
        print(f"Mean Silohoutte: {avg_s:.2f}, Interval: {silohoutte_interval:.3f}, Confidence Interval ({s_cf[0]:.3f}, {s_cf[1]:.3f})")
        print()
    
    
    ks = range(2, max_K + 1)

    plt.figure()
    plt.errorbar(ks, avg_loss, yerr = l_i, capsize=5, label="Loss", fmt='-o')
    plt.title(f"Elbow Method (Loss) for {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(f"{dataset_name}_loss_plot.png")
    

    plt.figure()
    plt.errorbar(ks, avg_silohoutte, yerr = s_i, capsize=5, label="Silhouette", fmt='-o')
    plt.title(f"Elbow Method (Silhouette) for {dataset_name}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.legend()
    plt.savefig(f"{dataset_name}_silhouette_plot.png")



find_elbow_and_plot(dataset1, "Dataset 1")
print()
find_elbow_and_plot(dataset2, "Dataset 2")
