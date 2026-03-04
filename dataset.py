import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FusionDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== Image =====
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        # ===== GPS (dx, dy 已标准化版本) =====
        geo = torch.tensor(
            [row["dx"], row["dy"]],
            dtype=torch.float32
        )

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, geo, label