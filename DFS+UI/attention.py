import torch
import torch.nn.functional as F
from encoder import MobileNetV2_encoder
from decoder import LRASPPV2

class Attention_pooling(torch.nn.Module):
    def __init__(self, pretrained=True, mask_learn_rate=0.5):
        super(Attention_pooling, self).__init__()
        self.encoder = MobileNetV2_encoder(pretrained)
        self.decoder = LRASPPV2()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(1280, 512, True),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64, True),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, input):
        low_fea, high_fea = self.encoder(input)
        mask = self.decoder(high_fea, low_fea)
        att_mask = torch.unsqueeze(torch.softmax(mask, 1)[:, 1, :, :], 1)
        pred = torch.sum(high_fea * att_mask, dim=(2, 3)) / (torch.sum(att_mask, dim=(2, 3)) + 1e-8)
        pred = pred.view(pred.size(0), -1)
        out_0 = self.linear(pred)
        return pred