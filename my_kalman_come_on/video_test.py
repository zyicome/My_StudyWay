import new_kalman
import robots
import detection
import robot_sort
import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO("E:/pythonProject/YOLOv8_RobotMasterTracking/runs/detect/train17/weights/best.pt")
model_armor = YOLO("E:/pythonProject/YOLOv8_RobotMasterTracking/runs/detect/train13/weights/best.pt")

video_path = "C:/Users/zyb/Desktop/OpenCV_Pictureandfile/RobotMaster_video/radar_2.mp4"

init_measurement = np.array([[0], [0], [0], [0]])
init_robot = robot_sort.Robot(init_measurement)
All_Robots = [init_robot, init_robot, init_robot, init_robot, init_robot, init_robot, init_robot]
cap = cv2.VideoCapture(video_path)

# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()
    frame = cv2.resize(frame, (1024, 628))
    if success:
        # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
        results = model(frame)

        annotated_frame = results[0].plot()

        for detection_result in results:
            detection_boxes = detection_result.boxes
            detection_confidence = detection_boxes.conf.cpu().numpy().astype(float)
            detection_xyxy = detection_boxes.xyxy.cpu().numpy().astype(int)
            for i in range(len(detection_xyxy)):
                detection_xyxy[i][2] = detection_xyxy[i][2] - detection_xyxy[i][0]
                detection_xyxy[i][3] = detection_xyxy[i][3] - detection_xyxy[i][1]
            detection_tlwh = detection_xyxy
            detection_feature = detection_boxes.cls.cpu().numpy().astype(int)
            detections = []
            for detection_id in range(len(detection_tlwh)):
                new_detection = detection.Detection(detection_tlwh[detection_id], detection_confidence[detection_id],
                                                    detection_feature[detection_id])
                detections.append(new_detection)
            '''print("tlwh", detection_tlwh)
            print("confidence", detection_confidence)
            print("feature", detection_feature)
            print("1", detections[1].tlwh[1])
            print(detections)
            kuang = frame[int(detections[0].tlwh[1]):int((detections[0].tlwh[1] + detections[0].tlwh[3])),
                    int(detections[0].tlwh[0]):int((detections[0].tlwh[0] + detections[0].tlwh[2]))]
            kuang = cv2.resize(kuang, (300, 300))
            cv2.imshow("1", kuang)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)'''
            Robotor = robots.Robotor(All_Robots, frame)
            Robotor.predict()
            All_Robots = Robotor.update(detections)
            for robot in All_Robots:
                if robot.Armor_Confidence == 0:
                    continue
                frame = robot.draw_mean(frame)
                # cv2.imshow("frame", frame)
                # cv2.waitKey(0)
            # 展示带注释的帧
            cv2.imshow("frame", frame)
            # 如果按下'q'则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            else:
                # 如果视频结束则退出循环
                break

        # 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
