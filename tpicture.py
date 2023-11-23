import cv2
import os


file_path = r'E:/pythonProject/MyWriteMnist/train/train_p'
file_label = [cla for cla in os.listdir(file_path)]

for cla in file_label:
    img_path = os.path.join(file_path, cla)
    img_i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.resize(img_i, (224, 224))
    os.remove(img_path)
    cv2.imwrite(img_path, img_b)
    if img_b.ndim == 2:
        print('图像是单通道灰度图')
    else:
        print('图像包含多个通道')

