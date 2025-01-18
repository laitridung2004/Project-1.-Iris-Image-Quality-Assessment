import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2_encoder(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2_encoder, self).__init__()
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.low_feature = model.features[:5]
        self.high_feature = model.features[5:]

    def forward(self, input):
        out1 = self.low_feature(input)
        out2 = self.high_feature(out1)
        return out1, out2