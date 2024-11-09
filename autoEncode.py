import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import perf_counter
from tqdm import tqdm, trange
from line_profiler import profile
import random

from data import CustomDataset, getDataset

profile.disable()


class MixedAutoencoder(nn.Module):
    def __init__(
        self,
        categorical_dims,
        continuous_dim,
        embedding_dim,
        hidden_dims=[64, 32],
        latent_dim=16,
    ):
        """
        categorical_dims: list of integers representing number of categories for each categorical feature
        continuous_dim: number of continuous features
        embedding_dims: list of embedding dimensions for each categorical feature (if None, will use rule of thumb)
        hidden_dims: list of hidden layer dimensions
        latent_dim: dimension of the latent space
        """
        super().__init__()

        self.categorical_dims = categorical_dims
        self.numCategoricalInputs = len(categorical_dims)
        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim

        # Create embedding layers
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, embedding_dim) for n_cat in categorical_dims]
        )

        # Calculate total input dimension after embeddings
        total_embedding_dim = embedding_dim * len(categorical_dims)
        input_dim = total_embedding_dim + continuous_dim

        # Encoder layers
        print(f"Encoder Layers:")
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            print(f"  {current_dim}->{hidden_dim}")
            current_dim = hidden_dim
        print(f"  {current_dim}->{latent_dim}")

        # Latent space projection
        encoder_layers.append(nn.Linear(current_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            current_dim = hidden_dim

        self.decoder_shared = nn.Sequential(*decoder_layers)

        # Separate outputs for categorical and continuous features
        self.categorical_outputs = nn.ModuleList(
            [nn.Linear(hidden_dims[0], dim) for dim in categorical_dims]
        )
        self.continuous_output = nn.Linear(hidden_dims[0], continuous_dim)

    @profile
    def encode(self, categorical_inputs, continuous_input):
        # Process categorical inputs through embeddings
        embedded = []
        for i in range(self.numCategoricalInputs):
            embedded.append(self.embeddings[i](categorical_inputs[:, i]))
        embedded = torch.cat(embedded, dim=1)

        # Concatenate with continuous input
        x = torch.cat([embedded, continuous_input], dim=1)

        # Encode
        latent = self.encoder(x)
        return latent

    @profile
    def forward(self, categorical_inputs, continuous_input):
        # Encode
        latent = self.encode(categorical_inputs, continuous_input)

        # Decode
        decoded = self.decoder_shared(latent)

        # Generate outputs
        cat_outputs = [output(decoded) for output in self.categorical_outputs]
        cont_output = self.continuous_output(decoded)

        return cat_outputs, cont_output


@profile
def trainEncoder():
    for epoch in range(numEpochs):
        total_loss = 0
        dataset.shuffle()
        numBatches = dataset.numBatches(batchSize)
        start = perf_counter()
        for cat_inputs, cont_inputs in tqdm(
            dataset.iter(batchSize), desc=f"Epoch {epoch+1}", total=numBatches
        ):
            # Move inputs to device
            cat_inputs = cat_inputs.to(device)
            cont_inputs = cont_inputs.to(device)

            # Forward pass
            cat_outputs, cont_output = model(cat_inputs, cont_inputs)

            # Calculate losses
            cat_loss = 0.0
            for i in range(len(cat_outputs)):
                cat_loss += cat_criterion(cat_outputs[i], cat_inputs[:, i])
            cont_loss = cont_criterion(cont_output, cont_inputs)

            # Combine losses
            loss = cat_loss / len(cat_outputs) + cont_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = perf_counter() - start
        avg_loss = total_loss / numBatches

        print(
            f"Epoch [{epoch+1}/{numEpochs}], Average Loss: {avg_loss:.4f}, {int(numDataRows/elapsed):,} rows/sec"
        )

        torch.save(model, "model.pt")


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


# Example usage:
if __name__ == "__main__":
    # Settings
    learningRate = 1e-3
    batchSize = 2**16
    numEpochs = 30
    numDataRows = 5_000_000

    hiddenDims = [64, 32]
    embeddingDim = 8
    latentDim = 2

    numClusters = 5
    numClusteringIters = 100
    numClusteringRenderPoints = 1000

    print("Settings:")
    print(f"  Learning Rate: {learningRate}")
    print(f"  Batch Size: {batchSize}")
    print(f"  # Epochs: {numEpochs}")
    print(f"  Embedding Dim: {embeddingDim}")
    print(f"  Latent Dim: {latentDim}")
    print(f"  # Clusters: {numClusters}")
    print(f"  # Clustering Iters: {numClusteringIters}")

    dataset = getDataset(numDataRows)

    print(f"{numDataRows:,} rows of data")

    # Initialize model
    model = MixedAutoencoder(
        categorical_dims=dataset.categoricalDims,  # Number of categories for each categorical feature
        continuous_dim=dataset.continuousDim,  # Number of continuous features
        embedding_dim=embeddingDim,  # Embedding dimension
        hidden_dims=hiddenDims,  # Hidden layer dimensions
        latent_dim=latentDim,  # Latent space dimension
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    cat_criterion = nn.CrossEntropyLoss()
    cont_criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = model.to(device)

    trainEncoder()

    print(f"Finished")
