import pickle
import matplotlib.pyplot as plt
from pca import PCA
from autoencoder import AutoEncoder
from sklearn.manifold import TSNE
from umap import UMAP 

# Load datasets
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))


def plot_2d(data, title, x, y):
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(f"{title}_loss_plot.png")


def apply_pca(dataset, title):
    pca = PCA(projection_dim=2)
    pca.fit(dataset)
    reduced_data = pca.transform(dataset)
    plot_2d(reduced_data, title=f"PCA: {title}", x="PC1", y="PC2")


def apply_autoencoder(dataset, title):
    input_dim = dataset.shape[1]
    projection_dim = 2
    learning_rate = 0.01
    iteration_count = 1000

    autoencoder = AutoEncoder(input_dim, projection_dim, learning_rate=learning_rate, iteration_count=iteration_count)
    autoencoder.fit(dataset)
    reduced_data = autoencoder.transform(dataset)
    plot_2d(reduced_data, title=f"Autoencoder: {title}", x="x", y="y")


def apply_tsne(dataset, title):
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(dataset)
    plot_2d(reduced_data, f"t-SNE: {title}", x="x", y="y")


def apply_umap(dataset, title):
    umap_model = UMAP(n_neighbors=10, n_components=2)
    reduced_data = umap_model.fit_transform(dataset)
    plot_2d(reduced_data, f"UMAP: {title}", x="x", y="y")


apply_pca(dataset1, "Dataset 1")
apply_pca(dataset2, "Dataset 2")

apply_autoencoder(dataset1, "Dataset 1")
apply_autoencoder(dataset2, "Dataset 2")

apply_tsne(dataset1, "Dataset 1")
apply_tsne(dataset2, "Dataset 2")

apply_umap(dataset1, "Dataset1")
apply_umap(dataset2, "Dataset2")

