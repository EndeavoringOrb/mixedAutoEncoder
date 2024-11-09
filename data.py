import numpy as np
import torch


class CustomDataset:
    def __init__(
        self, categorical_data, continuous_data, categoricalDims, continuousDim, device
    ):
        """
        categorical_data: list of numpy arrays, one for each categorical feature
        continuous_data: numpy array of continuous features
        """
        self.categoricalDims = categoricalDims
        self.continuousDim = continuousDim
        self.categorical_data = [
            torch.tensor(cat, dtype=torch.int8) for cat in categorical_data
        ]
        self.categorical_data = torch.stack(self.categorical_data, dim=-1).to(device)
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32).to(device)
        self.length = self.categorical_data.shape[0]

    def shuffle(self):
        r = torch.randperm(self.length)
        self.categorical_data = self.categorical_data[r]
        self.continuous_data = self.continuous_data[r]

    def __len__(self):
        return self.length

    def iter(self, batchSize):
        idx = 0
        while idx + batchSize <= self.length:
            catData = self.categorical_data[idx : idx + batchSize].to(dtype=torch.int64)
            contData = self.continuous_data[idx : idx + batchSize]
            idx += batchSize
            yield catData, contData
        if idx < self.length:
            catData = self.categorical_data[idx:].to(dtype=torch.int64)
            contData = self.continuous_data[idx:]
            yield catData, contData

    def numBatches(self, batchSize):
        return int(np.ceil(self.length / batchSize))


def getDataset(numRows, device):
    categoricalDims = [6, 4, 3, 16, 4, 6]  # categorical
    categoricalDims += [2] * 10  # boolean
    continuousDim = 30  # continuous

    # Example data
    categoricalData = [
        np.random.randint(0, dim, size=numRows) for dim in categoricalDims
    ]
    continuousData = np.random.randn(numRows, continuousDim)

    # Create dataset and dataloader
    dataset: CustomDataset = CustomDataset(
        categoricalData, continuousData, categoricalDims, continuousDim, device
    )

    return dataset
