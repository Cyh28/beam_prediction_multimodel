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
    def __init__(self, num_classes=64, backbone="resnet34"):
        super().__init__()

        if backbone == "resnet34":
            self.cnn = models.resnet34(pretrained=True)
            self.cnn.fc = nn.Identity()
            img_dim = 512

        elif backbone == "vit":
            self.cnn = models.vit_b_16(pretrained=True)
            self.cnn.heads = nn.Identity()
            img_dim = 768

        elif backbone == "swin":
            self.cnn = models.swin_t(pretrained=True)
            self.cnn.head = nn.Identity()
            img_dim = 768

        else:
            raise ValueError("Unknown backbone")

        self.geo = GeoBranch()

        self.fusion = nn.Sequential(
            nn.Linear(img_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, geo):
        img_feat = self.cnn(image)
        geo_feat = self.geo(geo)

        fused = torch.cat([img_feat, geo_feat], dim=1)
        return self.fusion(fused)