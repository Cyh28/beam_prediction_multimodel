import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class FusionDataset(Dataset):
    """
    多模态融合数据集
    支持 image + gps(dx,dy) + radar + label
    """

    def __init__(
        self,
        csv_path,
        mode="train",
        image_size=224,
        strong_aug=True,
        radar_feature="ra_map"
    ):

        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.image_size = image_size
        self.radar_feature = radar_feature

        valid_radar_features = {"ra_map", "rd_map", "range_fft"}
        if self.radar_feature not in valid_radar_features:
            raise ValueError(
                f"radar_feature 必须是 {sorted(valid_radar_features)}，当前是: {self.radar_feature}"
            )

        # ===== ImageNet 归一化 =====
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if mode == "train":

            if strong_aug:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05
                    ),
                    transforms.ToTensor(),
                    normalize,
                    transforms.RandomErasing(p=0.25)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize
                ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize
            ])

        # ===== 数据检查 =====
        required_columns = [
            "image_path",
            "radar_path",
            "dx",
            "dy",
            "label"
        ]

        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CSV 中缺少必要列: {col}")

    def __len__(self):
        return len(self.df)

    def load_radar(self, radar_path):
        """
        读取 radar npz
        """

        if not os.path.exists(radar_path):
            raise FileNotFoundError(f"Radar 文件不存在: {radar_path}")

        radar_data = np.load(radar_path)

        radar_map = radar_data[self.radar_feature]

        # 对 3D 雷达特征在最后一维做平均，得到 2D 输入。
        if radar_map.ndim == 3:
            radar_map = np.mean(radar_map, axis=2)
        elif radar_map.ndim != 2:
            raise ValueError(
                f"不支持的雷达特征维度: {radar_map.ndim}，路径: {radar_path}"
            )

        # 标准化
        radar_map = (radar_map - radar_map.mean()) / (radar_map.std() + 1e-6)

        radar_tensor = torch.tensor(
            radar_map,
            dtype=torch.float32
        ).unsqueeze(0)

        return radar_tensor

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ===== Image =====
        img_path = row["image_path"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像路径不存在: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # ===== Radar =====
        radar_path = row["radar_path"]
        radar = self.load_radar(radar_path)

        # ===== GPS =====
        geo = torch.tensor(
            [row["dx"], row["dy"]],
            dtype=torch.float32
        )

        # ===== Label =====
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return image, geo, radar, label