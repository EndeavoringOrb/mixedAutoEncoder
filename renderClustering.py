from autoEncode import CustomDataset, MixedAutoencoder
from data import CustomDataset, getDataset
from tqdm import tqdm, trange
import random
import torch


@torch.no_grad()
def trainClustersRender():
    # Get all points
    numBatches = dataset.numBatches(batchSize)
    points = []
    for cat_inputs, cont_inputs in tqdm(
        dataset.iter(batchSize), desc=f"Getting Points", total=numBatches
    ):
        # Move inputs to device
        cat_inputs = cat_inputs.to(device)
        cont_inputs = cont_inputs.to(device)

        # Forward pass
        out = model.encode(cat_inputs, cont_inputs)
        points.append(out)
    points = torch.cat(points)

    # For visualization: Use PCA to reduce dimensionality to 2D if needed
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Perform PCA if dimensions > 2
    if points.shape[1] > 2:
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points.cpu().numpy())
        points_2d = torch.tensor(points_2d, device=device)
    else:
        points_2d = points

    # Parameters for visualization
    num_sample_points = min(
        numClusteringRenderPoints, len(points)
    )  # Adjust based on your needs
    sample_indices = torch.randperm(len(points))[:num_sample_points]

    # Initialize centroids randomly
    startPointIndices = random.sample(range(len(points)), numClusters)
    centroids = points[startPointIndices]
    if points.shape[1] > 2:
        centroids_2d = torch.tensor(
            pca.transform(centroids.cpu().numpy()), device=device
        )
    else:
        centroids_2d = centroids

    # Create output directory for plots
    import os

    os.makedirs("clustering_progress", exist_ok=True)

    # Plot initial state
    plt.figure(figsize=(10, 10))
    plt.scatter(
        points_2d[sample_indices, 0].cpu(),
        points_2d[sample_indices, 1].cpu(),
        c="gray",
        alpha=0.5,
        s=50,
    )
    plt.scatter(
        centroids_2d[:, 0].cpu(),
        centroids_2d[:, 1].cpu(),
        c="red",
        marker="x",
        s=200,
        linewidths=3,
    )
    plt.title(f"Clustering Progress - Initial State")
    plt.savefig(f"clustering_progress/cluster_iter_0.png")
    plt.close()

    for iter_num in trange(numClusteringIters, desc="Clustering Data"):
        # Calculate distances from data points to centroids
        distances = torch.cdist(points, centroids)

        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)

        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(numClusters):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(points[labels == i], dim=0)

        # Update 2D centroids for visualization
        if points.shape[1] > 2:
            centroids_2d = torch.tensor(
                pca.transform(centroids.cpu().numpy()), device=device
            )

        # Plot current state
        plt.figure(figsize=(10, 10))

        # Plot sample points colored by their cluster
        scatter = plt.scatter(
            points_2d[sample_indices, 0].cpu(),
            points_2d[sample_indices, 1].cpu(),
            c=labels[sample_indices].cpu(),
            cmap="tab10",
            alpha=0.5,
            s=50,
        )

        # Plot centroids
        plt.scatter(
            centroids_2d[:, 0].cpu(),
            centroids_2d[:, 1].cpu(),
            c="red",
            marker="x",
            s=200,
            linewidths=3,
        )

        plt.title(f"Clustering Progress - Iteration {iter_num + 1}")
        plt.savefig(f"clustering_progress/cluster_iter_{iter_num + 1}.png")
        plt.close()

    return centroids, labels


# Optional: Create a GIF from the saved images
def create_animation():
    import imageio
    import glob

    # Get list of files in correct order
    files = sorted(
        glob.glob("clustering_progress/cluster_iter_*.png"),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # Create GIF
    images = [imageio.imread(file) for file in files]
    imageio.mimsave(
        "clustering_animation.gif", images, duration=0.5
    )  # 0.5 seconds per frame


if __name__ == "__main__":
    batchSize = 2**16
    numDataRows = 5_000_000
    numClusters = 10
    numClusteringIters = 100
    numClusteringRenderPoints = 1000

    dataset = getDataset(numDataRows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: MixedAutoencoder = torch.load("model.pt", weights_only=False)

    trainClustersRender()
    create_animation()
