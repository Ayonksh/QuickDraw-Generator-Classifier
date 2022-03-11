import torch
import torch.nn as nn
import torchvision.models as models

def resnet34(num_class, pretrained = False):
    model = models.resnet34(pretrained)
    conv1_out_channels = model.conv1.out_channels
    model.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size = 3,
                            stride = 1, padding = 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 2)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_class)
    return model

class QuickDrawNet(nn.Module):
    def __init__(self, num_class):
        super(QuickDrawNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x