import time

import PIL.Image
import cv2
import os

import torch.cuda
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

from mynet import MyNet

predict_path = 'E:/pythonProject/MyWriteMnist/data_predict'
flower_class = [cla for cla in os.listdir(predict_path)]

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_img(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    ret, img_two = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    img_tensor = transform(img_two)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    return img_tensor, img_two


def get_model(path_state_dict, vis_model=False):
    model = MyNet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(1, 224, 224), device="cpu")

    model.to(device)
    return model


if __name__ == "__main__":
    num_class = 10
    for cla in flower_class:
        file_path = os.path.join(predict_path, cla)
        img_tensor, img = process_img(file_path)

        model = get_model('E:/pythonProject/MyWriteMnist/My_model/model_pth')

        with torch.no_grad():
            time_tic = time.time()
            outputs = model(img_tensor)
            print(outputs.data)
            time_toc = time.time()

        _, prod_int = torch.max(outputs.data, 1)
        _, top_int = torch.topk(outputs.data, 1, dim=1)

        prod_idx = int(prod_int.cpu().numpy())
        print("img: {} is: {}".format(os.path.basename(file_path), prod_int))
        plt.imshow(img)
        plt.title("predict:{}".format(prod_idx))
        plt.text(5, 45, "top {}:{}".format(cla, prod_idx), bbox=dict(fc='yellow'))
        plt.show()
        cv2.waitKey(0)
