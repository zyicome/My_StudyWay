import torch
from torch import nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(torch.nn.Conv2d(1, 6, 3, 1)
                                   , nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(torch.nn.Conv2d(6, 16, 5),
                                   nn.ReLU(), nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(torch.nn.Conv2d(16, 64, 5, stride=2),
                                   nn.ReLU(), nn.MaxPool2d(3, 2))
        self.fc1 = nn.Sequential(torch.nn.Linear(12 * 12 * 64, out_features=84), nn.BatchNorm1d(84), nn.ReLU())
        self.fc2 = nn.Sequential(torch.nn.Linear(in_features=84, out_features=24), nn.BatchNorm1d(24),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(torch.nn.Linear(in_features=24, out_features=10), nn.BatchNorm1d(10),
                                 nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = MyNet()

