"""Stage 1 frame classifier and stage 2 temporal 1D CNN."""
import timm
import torch.nn as nn
import torch.nn.functional as F

from dfl import config


class net(nn.Module):
    """ImageNet-pretrained timm backbone with a new 10-class head."""

    def __init__(self, modelname=config.MODEL_NAME, out_dim=config.NUM_CLASSES, freeze_bn=True):
        super(net, self).__init__()
        self.model = timm.create_model(modelname, pretrained=True)
        self.model.reset_classifier(out_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class cnn(nn.Module):
    """1D CNN over a window of 10-class frame scores.

    Input shape: (batch, 10, 2 * STEPS + 1). Output: 10 class logits for
    the center frame.
    """

    def __init__(self):
        super(cnn, self).__init__()
        hdn = 32
        self.fc = nn.Conv1d(10, hdn, 13, bias=False, stride=2)
        self.bn = nn.BatchNorm1d(hdn)
        self.do = nn.Dropout(0.2)
        self.fc2 = nn.Conv1d(hdn, hdn * 2, 7, bias=False, stride=2)
        self.bn2 = nn.BatchNorm1d(hdn * 2)
        self.do2 = nn.Dropout(0.25)
        self.fc3 = nn.Conv1d(hdn * 2, hdn * 2, 5, bias=False, stride=1)
        self.bn3 = nn.BatchNorm1d(hdn * 2)
        self.do3 = nn.Dropout(0.25)
        self.fc4 = nn.Conv1d(hdn * 2, hdn * 2, 3, bias=False, stride=1)
        self.bn4 = nn.BatchNorm1d(hdn * 2)
        self.do4 = nn.Dropout(0.25)
        self.fc5 = nn.Linear(hdn * 2, 10)

    def forward(self, x):
        x = self.do(F.relu(self.bn(self.fc(x))))
        x = self.do2(F.relu(self.bn2(self.fc2(x))))
        x = self.do3(F.relu(self.bn3(self.fc3(x))))
        x = self.do4(F.relu(self.bn4(self.fc4(x))))
        return self.fc5(F.adaptive_avg_pool1d(x, 1).squeeze(-1))
