import robot_sort
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("E:/pythonProject/YOLOv8_RobotMasterTracking/runs/detect/train17/weights/best.pt")
model_armor = YOLO("E:/pythonProject/YOLOv8_RobotMasterTracking/runs/detect/train13/weights/best.pt")

img_init = cv2.imread(
    "G:/Robomaster_data/DJI ROCO/DJI ROCO/robomaster_North China Regional Competition/image/SPRVsTOE_BO5_3_50.jpg"
    , cv2.IMREAD_COLOR)
img_init = cv2.resize(img_init, (1920, 1080))
print(img_init.shape)
result = model.predict(img_init)
img = result[0].plot(labels=False)
img = cv2.resize(img, (1920, 1080))
cv2.imshow("img", img)
cv2.waitKey(0)
for i in result:
    boxes = i.boxes
    class_id = boxes.cls.cpu().numpy().astype(int)
    box = boxes.xyxy.cpu().numpy().astype(int)
    print(class_id)
    print(box)
    for j in range(len(box)):
        h = box[j][3] - box[j][1]
        a = (box[j][2] - box[j][0]) / h
        x = (box[j][2] + box[j][0]) / 2
        y = (box[j][3] + box[j][1]) / 2
        measurement = np.array([[x], [y], [a], [h]])
        robot_1 = robot_sort.Robot(measurement)
        robot_1.get_mean_cov(measurement)
        picture = robot_1.draw_mean(img_init)
        cv2.imshow("picture", picture)
        cv2.waitKey(0)
        measurement = np.array([[x + 30], [y], [a], [h]])
        robot_1.get_mean_cov(measurement)
        picture = robot_1.draw_mean(img_init)
        cv2.imshow("picture", picture)
        cv2.waitKey(0)
        measurement = np.array([[x + 55], [y], [a], [h]])
        robot_1.get_mean_cov(measurement)
        picture = robot_1.draw_mean(img_init)
        cv2.imshow("picture", picture)
        cv2.waitKey(0)
        measurement = np.array([[x + 80], [y], [a], [h]])
        robot_1.get_mean_cov(measurement)
        picture_0 = robot_1.draw_mean(img_init)
        cv2.imshow("picture", picture_0)
        cv2.waitKey(0)
        robot_1.get_mean_cov()
        picture_1 = robot_1.draw_mean(img_init)
        cv2.imshow("picture", picture_1)
        cv2.waitKey(0)
        robot_1.get_mean_cov()
        picture_2 = robot_1.draw_mean(img_init)
        cv2.imshow("picture", picture_2)
        cv2.waitKey(0)

