import pickle
from Distance import Distance
from Knn import KNN


import matplotlib.pyplot as plt


# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))
print(dataset)
print(labels)



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each class with a different color
unique_labels = set(labels)
for label in unique_labels:
    points = dataset[labels == label]  # Assuming labels are numpy-compatible
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Class {label}")

# Add plot details
ax.set_title("3D Dataset Visualization")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.legend()
plt.savefig("plot.png")
print("Plot saved as 'plot.png'")
