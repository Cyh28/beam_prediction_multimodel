import torch
import torch.nn as nn
import torchvision.models as models


class GeoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, num_classes=64):
        super().__init__()

        # ===== Image branch =====
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # 输出 512 维

        # ===== Geo branch =====
        self.geo = GeoBranch()

        # ===== Fusion head =====
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, geo):
        img_feat = self.cnn(image)      # (B,512)
        geo_feat = self.geo(geo)        # (B,128)

        fused = torch.cat([img_feat, geo_feat], dim=1)
        out = self.fusion(fused)

        return out