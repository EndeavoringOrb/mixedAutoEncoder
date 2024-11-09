from data import CustomDataset, getDataset
from time import perf_counter
from tqdm import tqdm, trange
import torch.optim as optim
import torch.nn as nn
import torch
import os


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

    def forward(self, categorical_inputs, continuous_input):
        # Encode
        latent = self.encode(categorical_inputs, continuous_input)

        # Decode
        decoded = self.decoder_shared(latent)

        # Generate outputs
        cat_outputs = [output(decoded) for output in self.categorical_outputs]
        cont_output = self.continuous_output(decoded)

        return cat_outputs, cont_output


def trainModel():
    # Get the number of the current training run
    runNumber = (
        max(int(item) for item in os.listdir("trainingRuns")) + 1
        if len(os.listdir("trainingRuns")) > 0
        else 0
    )
    os.makedirs(f"trainingRuns/{runNumber}/models")

    for epoch in range(numEpochs):
        total_loss = torch.tensor(0, dtype=torch.float32, device=device)
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
            cat_loss = torch.tensor(0, dtype=torch.float32, device=device)
            for i in range(len(cat_outputs)):
                cat_loss += cat_criterion(cat_outputs[i], cat_inputs[:, i])
            cont_loss = cont_criterion(cont_output, cont_inputs)

            # Combine losses
            loss = cat_loss / len(cat_outputs) + cont_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss

        elapsed = perf_counter() - start  # Get the amount of seconds this epoch took
        avg_loss = total_loss / numBatches  # Get the average loss for this epoch

        print(
            f"Epoch [{epoch+1}/{numEpochs}], Average Loss: {avg_loss:.4f}, {int(numDataRows/elapsed):,} rows/sec"
        )

        with open(f"trainingRuns/{runNumber}/loss.txt", "a", encoding="utf-8") as f:
            f.write(f"{avg_loss}\n")

        torch.save(model, f"trainingRuns/{runNumber}/models/E{epoch+1}.pt")


if __name__ == "__main__":
    # Settings
    learningRate = 1e-3
    batchSize = 2**16
    numEpochs = 100
    numDataRows = 2**20

    hiddenDims = [64, 32]
    embeddingDim = 8
    latentDim = 8

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

    # Init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Get data
    dataset: CustomDataset = getDataset(numDataRows, device)
    print(f"{numDataRows:,} rows of data")

    # Initialize model
    model = MixedAutoencoder(
        categorical_dims=dataset.categoricalDims,  # Number of categories for each categorical feature
        continuous_dim=dataset.continuousDim,  # Number of continuous features
        embedding_dim=embeddingDim,  # Embedding dimension
        hidden_dims=hiddenDims,  # Hidden layer dimensions
        latent_dim=latentDim,  # Latent space dimension
    ).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    cat_criterion = nn.CrossEntropyLoss()
    cont_criterion = nn.MSELoss()

    # Train model
    trainModel()

    print(f"Finished")
