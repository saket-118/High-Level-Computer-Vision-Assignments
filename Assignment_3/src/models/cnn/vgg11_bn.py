# src/models/cnn/vgg11_bn.py
import torch
import torch.nn as nn
from torchvision.models import vgg11_bn, VGG11_BN_Weights

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    """
    VGG-11-BN backbone + 2-layer MLP head.
    Q4-a  ➜  backbone frozen (features only, no classifier)
    Q4-b  ➜  optionally fine‐tune when fine_tune=True
    """

    def __init__(self,
                 layer_config,   # a list like [512, 256]
                 num_classes,    # 10 for CIFAR-10
                 activation,     # typically nn.ReLU
                 norm_layer,     # typically nn.BatchNorm1d
                 fine_tune: bool,
                 weights="DEFAULT"):
        super(VGG11_bn, self).__init__()

        if weights == "DEFAULT":
            full_vgg = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        else:
            full_vgg = vgg11_bn(weights=None)

        self.features = full_vgg.features  # This is a nn.Sequential of conv+BN layers

        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

        h1, h2 = layer_config  # e.g. [512, 256]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # pools (512,7,7) → (512,1,1)
            nn.Flatten(),                  # → (B, 512)
            nn.Linear(512,   h1, bias=True),
            norm_layer(h1),
            activation(),
            nn.Linear(h1,    h2, bias=True),
            norm_layer(h2),
            activation(),
            nn.Linear(h2,    num_classes, bias=True),
        )

        # 5) Initialize the new head’s weights in the usual way
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 6) If fine_tune == True, unfreeze the features as well
        if fine_tune:
            for param in self.features.parameters():
                param.requires_grad = True

            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    def forward(self, x):

        if not any(p.requires_grad for p in self.features.parameters()):
            with torch.no_grad():
                out = self.features(x)   # (B, 512, 7, 7)
        else:
            # If we are fine-tuning, we do not wrap in no_grad
            out = self.features(x)

        out = self.head(out)             # (B, num_classes)
        return out
