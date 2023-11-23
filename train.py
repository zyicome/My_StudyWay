import torch
import os
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from mynet import MyNet
from matplotlib import pyplot as plt


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST("E:/mnist", True, transform=transform, download=True)
test_dataset = datasets.MNIST("E:/mnist", True, transform=transform, download=True)

b_size = 64

train_loader = DataLoader(train_dataset, b_size, True)
test_loader = DataLoader(test_dataset, b_size, False)


device = torch.device("cuda" if torch.cuda.is_available() else "cqu")

Learn_Speed = 0.01

Net = MyNet().to(device)

Loss = nn.CrossEntropyLoss()

deeper = optim.Adam(Net.parameters(), lr=Learn_Speed)


def down_loss(train_loss, test_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


t_loss = []
te_loss = []
epoch = 1
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
        t_loss.append(loss.item())
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
        # print("Predicated:", predicted)
        total += labels.size(0)
        loss = Loss(output_test, labels)
        correct += (predicted == labels).sum()
        te_loss.append(loss.item())
    print("correct:", correct)
    print("Test acc: {0}".format(correct.item() / total))
    model_file = 'My_model'
    if not os.path.exists(model_file):
        os.mkdir(model_file)
    torch.save(Net.state_dict(), 'My_model/model_pth')
    down_loss(t_loss, te_loss)



