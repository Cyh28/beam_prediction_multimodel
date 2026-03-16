import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class FusionDataset(Dataset):
    """
    多模态融合数据集
    支持 image + (dx, dy) + label

    适用于：
    - ResNet34
    - ViT
    - Swin Transformer
    """

    def __init__(
        self,
        csv_path,
        mode="train",
        image_size=224,
        strong_aug=True
    ):
        """
        Args:
            csv_path (str): CSV 文件路径
            mode (str): "train" / "val" / "test"
            image_size (int): 输入尺寸
            strong_aug (bool): 是否使用较强数据增强（小数据推荐 True）
        """

        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.image_size = image_size

        # ===== ImageNet 预训练标准归一化 =====
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # ===== 训练模式 =====
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

        # ===== 验证 / 测试模式 =====
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize
            ])

        # ===== 数据检查 =====
        required_columns = ["image_path", "dx", "dy", "label"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CSV 中缺少必要列: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== 读取图像 =====
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像路径不存在: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # ===== GPS 特征 =====
        geo = torch.tensor(
            [row["dx"], row["dy"]],
            dtype=torch.float32
        )

        # ===== 标签 =====
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return image, geo, label