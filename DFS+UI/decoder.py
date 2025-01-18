import torch
import torch.nn.functional as F

class LRASPPV2(torch.nn.Module):
    def __init__(self, nclass=2):
        super(LRASPPV2, self).__init__()
        self.b0 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 128, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True)
        )
        self.b1 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(1280, 128, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.project = torch.nn.Conv2d(128, nclass, 1)
        self.shortcut = torch.nn.Conv2d(32, nclass, 1)

    def forward(self, x, y):
        size = x.shape[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2
        x = self.project(x)
        y = self.shortcut(y)
        out = F.adaptive_avg_pool2d(y, size) + x
        return out