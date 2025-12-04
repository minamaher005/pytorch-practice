

import torch
from torch.utils.data import Dataset, DataLoader


class FraudDataset(Dataset):

    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.x)
    
    def __getitem__(self, idx):
        """Return a specific item by index."""
        return self.x[idx], self.y[idx]


def create_dataloader(X_tensor, y_tensor, batch_size=64, shuffle=True):
 
    dataset = FraudDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
