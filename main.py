import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

train_dataset = datasets.MNIST("E:/mnist", True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST("E:/mnist", True, transform=transforms.ToTensor(), download=True)

b_size = 64

train_loader = DataLoader(train_dataset, b_size, True)
test_loader = DataLoader(test_dataset, b_size, True)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(torch.nn.Conv2d(1, 6, 3, 1,
                                                   2, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(torch.nn.Conv2d(6, 16, 5),
                                   nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(torch.nn.Linear(16 * 5 * 5, out_features=120), nn.BatchNorm1d(120), nn.ReLU())
        self.fc2 = nn.Sequential(torch.nn.Linear(in_features=120, out_features=84), nn.BatchNorm1d(84),
                                 nn.ReLU(), nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cqu")

Learn_Speed = 0.01

Net = MyNet().to(device)

Loss = nn.CrossEntropyLoss()

deeper = optim.Adam(Net.parameters(), lr=Learn_Speed)


def image_show(images, labels):
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    cv2.imshow("Result", img)
    key_pressed = cv2.waitKey(0)

epoch = 1
if __name__ == "__main__":
    for epoch in range(epoch):
        print("GPU", torch.cuda.is_available())
        print("start to learn")
        sum_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            deeper.zero_grad()
            outputs = Net(inputs)
            loss = Loss(outputs, labels)
            loss.backward()
            deeper.step()
            sum_loss += loss.item()
            if i % 100 == 99:
                print("[%d,%d] loss:%.03f" % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0
        Net.eval()
        correct = 0
        total = 0
        for data_test in test_loader:
            _images, _labels = data_test
            images, labels = Variable(_images).cuda(), Variable(_labels).cuda()
            output_test = Net(images)
            _, predicted = torch.max(output_test, 1)
            print("Predicated:", predicted)
            image_show(_images, _labels)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("correct:", correct)
        print("Test acc: {0}".format(correct.item() / total))

