
import torch.nn as nn
import torchvision.models as models
import torch

def create_backbone(name='mobilenet_v2', pretrained=True):
    # Always defaulting to mobilenet_v2 for safety
    b = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
    feat = b.classifier[1].in_features
    modules = list(b.features)
    backbone = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1,1)))
    return backbone, feat

class MultiTaskNet(nn.Module):
    def __init__(self, n_classes=4, backbone_name='mobilenet_v2', pretrained=True, meta_dim=2, bottleneck_dim=512):
        super().__init__()
        self.backbone, in_features = create_backbone(backbone_name, pretrained)
        self.bottleneck = nn.Linear(in_features, bottleneck_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(bottleneck_dim,128), nn.ReLU(), nn.Linear(128, n_classes))
        self.regressor = nn.Sequential(nn.Linear(bottleneck_dim + meta_dim, 128), nn.ReLU(), nn.Linear(128,1))

    def forward(self, x, meta=None):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.relu(x)
        cls_logits = self.classifier(x)
        if meta is None:
            meta = x.new_zeros(x.size(0), 2)
        reg_in = torch.cat([x, meta.float()], dim=1)
        days = self.regressor(reg_in).squeeze(1)
        return cls_logits, days
