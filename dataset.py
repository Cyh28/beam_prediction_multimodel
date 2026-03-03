# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GeoDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dx = self.df.iloc[idx]["dx"]
        dy = self.df.iloc[idx]["dy"]
        label = int(self.df.iloc[idx]["label"])

        x = torch.tensor([dx, dy], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y