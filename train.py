import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import time
from tqdm import tqdm


start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断GPU是否可用
# 数据预处理 数据增强
transform = transforms.Compose([
    # 对图像进行随机的裁剪crop以后再resize成固定大小（224*224）
    transforms.RandomResizedCrop(224),
    # 随机旋转20度（顺时针和逆时针）
    transforms.RandomRotation(20),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 将数据转换为tensor
    transforms.ToTensor()
])

# 读取数据
root = 'E:/pythonProject/Vgg16_T_CatAndDog/Mydata'  # root是数据集目录
# 获取数据的路径，使用transform增强变化
train_dataset = datasets.ImageFolder(root + '/train', transform)
test_dataset = datasets.ImageFolder(root + '/test', transform)
# 导入数据
# 每个批次8个数据，打乱
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
# 类别名称
classes = train_dataset.classes
# 类别编号
classes_index = train_dataset.class_to_idx
print("类别名称", classes)
print("类别编号", classes_index)
# models.下有很多pytorch提供的训练好的模型
model = models.vgg16(pretrained=True)
# 我们主要是想调用vgg16的卷积层，全连接层自己定义，覆盖掉原来的
# 如果想只训练模型的全连接层（不想则注释掉这个for）
for param in model.parameters():
    param.requires_grad = False
# 构建新的全连接层
# 25088：卷阶层输入的是25088个神经元，中间100是自己定义的，输出类别数量2
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 100),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(100, 2)
                                       # 这里可以加softmax也可以不加
                                       )
model = model.to(device)  # 将模型发送到GPU上
print("使用GPU:", next(model.parameters()).device)  # 输出：cuda:0
LR = 0.0001
# 定义代价函数
entropy_loss = nn.CrossEntropyLoss()  # 损失函数
# 定义优化器
optimizer = optim.SGD(model.parameters(), LR, momentum=0.9)
print("开始训练~")


def train():
    model.train()
    for i, data in enumerate(tqdm(train_loader, desc='train:')):
        # 获得数据和对应的标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据发送到GPU上
        # 获得模型预测结果，（64，10）
        out = model(inputs)
        # 交叉熵代价函数out(batch,C),labels(batch)
        loss = entropy_loss(out, labels).to(device)  # 别忘了损失函数也要发到GPU
        # 梯度清0
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()
    end1 = time.time()
    print("训练结束 运行时间：{:.3f}分钟".format((end1 - start) / 60))


t_loss = []
t_acc = []
te_loss = []
te_acc = []


def test():
    model.eval()
    correct = 0
    for i, data in enumerate(tqdm(test_loader, desc='test:')):
        # 获得数据和对应的标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 获得模型预测结果
        out = model(inputs)
        # 获得最大值，以及最大值所在的位置
        _, predicted = torch.max(out, 1)
        # 预测正确的数量
        correct += (predicted == labels).sum()
        te_loss.append(1 - correct.item() / len(test_dataset))
        te_acc.append(correct.item() / len(test_dataset) * 100)
    print("Test acc: {:.2f}".format(correct.item() / len(test_dataset)))
    print("Test loss:{:.2f}".format(1 - correct.item() / len(test_dataset)))  # 损失率+准确率为1
    end3 = time.time()
    print("预测结束 运行结束：{:.3f}分钟".format((end3 - start) / 60))

    correct = 0
    for i, data in enumerate(tqdm(train_loader, desc='train:')):
        # 获得数据和对应的标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 获得模型预测结果
        out = model(inputs)
        # 获得最大值，以及最大值所在的位置
        _, predicted = torch.max(out, 1)
        # 预测正确的数量
        correct += (predicted == labels).sum()
        t_loss.append(1 - correct.item() / len(train_dataset))
        t_acc.append(correct.item() / len(train_dataset) * 100)
    print("Train acc: {:.2f}".format(correct.item() / len(train_dataset)))
    print("Train loss:{:.2f}".format(1 - correct.item() / len(train_dataset)))
    end4 = time.time()
    print("预测结束 运行结束：{:.3f}分钟".format((end4 - start) / 60))
    return t_loss, t_acc, te_loss, te_acc


def draw_loss(train_loss, test_loss):
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="test_loss")
    plt.legend(loc="best")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("训练集和测试集loss对比图")
    plt.show()


def draw_acc(train_acc, test_acc):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(loc="best")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("训练集和测试集acc对比图")
    plt.show()


for epoch in range(0, 10):
    print('epoch:', epoch)
    train()
    t_loss, t_acc, te_loss, te_acc = test()


foload = "My_model"
if not os.path.exists(foload):
    os.mkdir("My_model")
torch.save(model.state_dict(), 'My_model/model.pth')
end2 = time.time()
print("程序结束，程序运行时间：{:.3f}分钟".format((end2 - start) / 60))
draw_loss(t_loss, te_loss)
draw_acc(t_acc, te_acc)
