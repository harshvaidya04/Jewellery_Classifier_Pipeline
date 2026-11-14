import timm
import torch.nn as nn


class EfficientNetWithDropout(nn.Module):
    def __init__(self, model_name: str, num_classes: int, drop1: float = 0.4, drop2: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=drop1),
            nn.Linear(feat_dim, 512),
            nn.SiLU(),
            nn.Dropout(p=drop2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(model_name: str, num_classes: int, drop1: float = 0.4, drop2: float = 0.3) -> EfficientNetWithDropout:
    return EfficientNetWithDropout(model_name, num_classes, drop1, drop2)
