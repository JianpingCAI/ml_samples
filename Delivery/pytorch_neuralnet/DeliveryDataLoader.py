import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


class DatasetDelivery(Dataset):

    def __init__(self, df_X, df_y, transformToTensor=True):
        self.X = df_X.values
        self.y = df_y.values
        self.transform = transformToTensor

    def __len__(self):
        return np.size(self.X, 0)

    def __getitem__(self, index):
        features = self.X[index, :]
        target = self.y[index]

        if self.transform:
            features = torch.from_numpy(
                features.astype(np.float32))  # .float()
            target = torch.tensor([target.astype(np.float32)])

        return features, target
