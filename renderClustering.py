from autoEncode import CustomDataset, MixedAutoencoder
from data import CustomDataset, getDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import torch
import os


@torch.no_grad
def trainClusters():
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

    # Initialize centroids randomly
    startPointIndices = random.sample(range(len(points)), numClusters)
    centroids = points[startPointIndices]

    for _ in trange(numClusteringIters, desc="Clustering Data"):
        # Calculate distances from data points to centroids
        distances = torch.cdist(points, centroids)

        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)

        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(numClusters):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(points[labels == i], dim=0)

    return centroids, points


def renderKMeans(centroids, points):
    numSamplePoints = min(numClusteringRenderPoints, len(points))
    sampleIndices = random.sample(range(len(points)), numSamplePoints)

    # Perform PCA if dimensions > 2
    if points.shape[1] > 2:
        pca = PCA(n_components=2)
        pca = pca.fit(points.cpu().numpy())
        centroids_2d = pca.transform(centroids.cpu().numpy())
        centroids_2d = torch.tensor(centroids_2d, device=device)
        points_2d = pca.fit_transform(points[sampleIndices].cpu().numpy())
        points_2d = torch.tensor(points_2d, device=device)
    else:
        centroids_2d = centroids.cpu().numpy()
        points_2d = points[sampleIndices].cpu().numpy()

    os.makedirs(f"trainingRuns/{trainingRunNumber}/clusters", exist_ok=True)

    imageNum = (
        max(
            int(item.split(".")[0])
            for item in os.listdir(f"trainingRuns/{trainingRunNumber}/clusters")
        )
        + 1
        if len(os.listdir(f"trainingRuns/{trainingRunNumber}/clusters")) > 0
        else 0
    )

    # Plot initial state
    plt.figure(figsize=(10, 10))
    plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c="gray",
        alpha=0.5,
        s=50,
    )
    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="red",
        marker="x",
        s=200,
        linewidths=3,
    )
    plt.title(f"Clustering Progress - Epoch {imageNum + 1}")
    plt.savefig(f"trainingRuns/{trainingRunNumber}/clusters/{imageNum}.png")
    plt.close()


# Optional: Create a GIF from the saved images
def create_animation():
    import imageio
    import glob

    # Get list of files in correct order
    files = sorted(
        os.listdir(f"trainingRuns/{trainingRunNumber}/clusters"),
        key=lambda x: int(x.split(".")[0]),
    )

    # Create GIF
    images = [
        imageio.imread(f"trainingRuns/{trainingRunNumber}/clusters/{file}")
        for file in files
    ]
    imageio.mimsave(
        f"trainingRuns/{trainingRunNumber}/clusteringAnimation.gif",
        images,
        duration=0.5,
    )  # 0.5 seconds per frame


if __name__ == "__main__":
    trainingRunNumber = int(input("Enter the training run you want to render: "))

    batchSize = 2**16
    numDataRows = 5_000_000
    numClusters = 10
    numClusteringIters = 10
    numClusteringRenderPoints = 1000

    dataset = getDataset(numDataRows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelPaths = sorted(
        os.listdir(f"trainingRuns/{trainingRunNumber}/models"),
        key=lambda item: int(item.split(".")[0].strip("E")),
    )

    for i, path in enumerate(modelPaths):
        print(f"Rendering model {i+1}/{len(modelPaths)}")
        model: MixedAutoencoder = torch.load(
            f"trainingRuns/{trainingRunNumber}/models/{path}", weights_only=False
        ).to(device)

        centroids, points = trainClusters()
        renderKMeans(centroids, points)
    create_animation()
