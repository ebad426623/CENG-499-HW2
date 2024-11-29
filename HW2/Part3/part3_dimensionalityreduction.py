import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP 
import matplotlib.pyplot as plt

# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))


def plot_2d(data, title, x, y):
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(f"{title}_loss_plot.png")


def apply_pca(dataset, title):
    pca = PCA(2)
    reduced_data = pca.fit_transform(dataset)
    plot_2d(reduced_data, title=f"PCA: {title}", x="PC1", y="PC2")


def apply_tsne(dataset, title):
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(dataset)
    plot_2d(reduced_data, f"t-SNE: {title}", x="x", y="y")


def apply_umap(dataset, title):
    umap_model = UMAP(n_neighbors=10, n_components=2)
    reduced_data = umap_model.fit_transform(dataset)
    plot_2d(reduced_data, f"UMAP: {title}", x="x", y="y")


apply_pca(dataset, "Dataset")
apply_tsne(dataset, "Dataset")
apply_umap(dataset, "Dataset")