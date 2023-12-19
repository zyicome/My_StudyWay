import numpy as np
import new_kalman
import cv2


class Robot:
    def __init__(self, measurement, Armor_Confidence=0, Armor_id=None):
        self.kf = new_kalman.KalmanFilter()
        mean, cov = self.kf.initiate(measurement)
        self.mean = mean  # 包含位置、大小信息
        self.cov = cov
        self.measurement = None
        self.xywh = np.zeros((1, 4)).T

        self.Is_Have_Measurement = True

        self.Armor_Confidence = Armor_Confidence
        self.Armor_id = Armor_id
        self.Have_Matched_Num = 0

        self.Not_Matched_Num = 0

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    def update(self, measurement=None):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        if measurement is None:
            self.Is_Have_Measurement = False
            return self.mean, self.cov
        self.Is_Have_Measurement = True
        self.mean, self.cov = self.kf.update(self.mean, self.cov, measurement)
        self.change_from_xyah_to_xywh()
        return self.mean, self.cov

    def get_mean_cov(self, measurement=None):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        if measurement is None:
            self.Is_Have_Measurement = False
            return self.mean, self.cov
        self.Is_Have_Measurement = True
        self.mean, self.cov = self.kf.update(self.mean, self.cov, measurement)
        self.change_from_xyah_to_xywh()
        return self.mean, self.cov

    def change_from_xyah_to_xywh(self):
        self.xywh[2][0] = self.mean[2][0] * self.mean[3][0]  # w
        self.xywh[0][0] = self.mean[0][0] - self.xywh[2][0] / 2  # x
        self.xywh[1][0] = self.mean[1][0] - self.mean[3][0] / 2  # y
        self.xywh[3][0] = self.mean[3][0]  # h

    def draw_mean(self, picture):
        self.change_from_xyah_to_xywh()
        if self.Have_Matched_Num >= 5:
            if self.Is_Have_Measurement is True:
                cv2.rectangle(picture, (int(self.xywh[0]), int(self.xywh[1])),
                            (int(self.xywh[0] + self.xywh[2]), int(self.xywh[1] + self.xywh[3])),
                            color=(0, 0, 255), thickness=3)
                cv2.putText(picture, str(self.Armor_id), (int(self.xywh[0] + 30), int(self.xywh[1] + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
            else:
                cv2.rectangle(picture, (int(self.xywh[0]), int(self.xywh[1])),
                        (int(self.xywh[0] + self.xywh[2]), int(self.xywh[1] + self.xywh[3])),
                        color=(255, 255, 255), thickness=2)
                cv2.putText(picture, str(self.Armor_id), (int(self.xywh[0] + 30), int(self.xywh[1] + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
        return picture
