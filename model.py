import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _build_backbone(backbone, pretrained=True):
    if backbone == "resnet34":
        net = models.resnet34(pretrained=pretrained)
        net.fc = nn.Identity()
        out_dim = 512
        input_size = 224
    elif backbone == "vit":
        net = models.vit_b_16(pretrained=pretrained)
        net.heads = nn.Identity()
        out_dim = 768
        input_size = 224
    elif backbone == "swin":
        net = models.swin_t(pretrained=pretrained)
        net.head = nn.Identity()
        out_dim = 768
        input_size = 224
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return net, out_dim, input_size


def _adapt_first_layer_to_single_channel(model, backbone):
    # Keep pretrained semantics by averaging RGB filters into one channel.
    if backbone == "resnet34":
        old = model.conv1
        new = nn.Conv2d(
            in_channels=1,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        model.conv1 = new
    elif backbone == "vit":
        old = model.conv_proj
        new = nn.Conv2d(
            in_channels=1,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        model.conv_proj = new
    elif backbone == "swin":
        old = model.features[0][0]
        new = nn.Conv2d(
            in_channels=1,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.features[0][0] = new


class RadarBranch(nn.Module):
    def __init__(self, backbone="resnet34"):
        super().__init__()
        self.backbone_name = backbone
        self.encoder, self.out_dim, self.input_size = _build_backbone(backbone)
        _adapt_first_layer_to_single_channel(self.encoder, backbone)

    def forward(self, radar):
        # Radar input is [B, 1, H, W]. Resize to match vision backbone assumptions.
        radar = F.interpolate(
            radar,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.encoder(radar)


class FusionModel(nn.Module):
    def __init__(self, num_classes=64, backbone="resnet34", radar_backbone=None):
        super().__init__()

        self.cnn, img_dim, _ = _build_backbone(backbone)
        radar_backbone = radar_backbone or backbone
        self.radar = RadarBranch(backbone=radar_backbone)
        radar_dim = self.radar.out_dim

        self.geo = GeoBranch()

        self.fusion = nn.Sequential(
            nn.Linear(img_dim + 128 + radar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, geo, radar):
        img_feat = self.cnn(image)
        geo_feat = self.geo(geo)
        radar_feat = self.radar(radar)

        fused = torch.cat([img_feat, geo_feat, radar_feat], dim=1)
        return self.fusion(fused)