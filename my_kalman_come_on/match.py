import numpy as np
from ultralytics import YOLO
import robot_sort
import robots
import detection
import cv2

model = YOLO("E:/pythonProject/YOLOv8_RobotMasterTracking/runs/detect/train17/weights/best.pt")
model_armor = YOLO("E:/pythonProject/YOLOv8_RobotMasterTracking/runs/detect/train13/weights/best.pt")


class Matcher:
    def __init__(self, Robots, detections):
        self.robots = Robots
        self.condidates = []
        self.detections = detections

        self.robots_num = len(Robots)

        self.detections_num = len(detections)

        self.Is_Matched = []

        self.Not_Matched_Detections = []

        self.Not_Matched_Robots = []

    def iou_match(self):
        if len(self.detections) == 0:
            self.Not_Matched_Robots = self.robots
            return
        cost_matrix = self.iou_matrix()
        min_matrix = np.min(cost_matrix, axis=1)
        min_arg_matrix = np.argmin(cost_matrix, axis=1)
        '''print("cost_matrix", cost_matrix)
        print("min_matrix", min_matrix)
        print("min_arg_matrix", min_arg_matrix)'''
        for match_id in range(len(min_arg_matrix)):
            if min_matrix[match_id] == 1.:
                self.Not_Matched_Robots.append(self.robots[match_id])
                self.robots[match_id].Is_Have_Measurement = False
                self.robots[match_id].measurement = None
            if self.detections[min_arg_matrix[match_id]].Is_Used is True:
                if self.robots[match_id].Armor_Confidence >= self.robots[self.detections[min_arg_matrix[match_id]].Is_Used_Id].Armor_Confidence:
                    self.Is_Matched.append(self.detections[min_arg_matrix[match_id]])
                    self.detections[min_arg_matrix[match_id]].Is_Used = True
                    self.detections[min_arg_matrix[match_id]].Is_Used_Id = match_id
                    self.robots[self.detections[min_arg_matrix[match_id]].Is_Used_Id].Is_Have_Measurement = False
                    self.robots[match_id].Is_Have_Measurement = True
                    '''print("match_id", match_id)
                    print("detection", self.detections)'''
                    xyah = np.array([[self.detections[min_arg_matrix[match_id]].to_xyah()[0]],
                                     [self.detections[min_arg_matrix[match_id]].to_xyah()[1]],
                                     [self.detections[min_arg_matrix[match_id]].to_xyah()[2]],
                                     [self.detections[min_arg_matrix[match_id]].to_xyah()[3]]])
                    self.robots[match_id].measurement = xyah
                    self.robots[match_id].Have_Matched_Num += 1
                else:
                    continue
            else:
                self.Is_Matched.append(self.detections[min_arg_matrix[match_id]])
                self.detections[min_arg_matrix[match_id]].Is_Used = True
                self.detections[min_arg_matrix[match_id]].Is_Used_Id = match_id
                self.robots[match_id].Is_Have_Measurement = True
                '''print("match_id", match_id)
                print("detection", self.detections)'''
                xyah = np.array([[self.detections[min_arg_matrix[match_id]].to_xyah()[0]],
                                 [self.detections[min_arg_matrix[match_id]].to_xyah()[1]],
                                 [self.detections[min_arg_matrix[match_id]].to_xyah()[2]],
                                 [self.detections[min_arg_matrix[match_id]].to_xyah()[3]]])
                self.robots[match_id].measurement = xyah
                self.robots[match_id].Have_Matched_Num += 1
        for match_id in range(len(min_arg_matrix)):
            np.delete(self.detections, min_arg_matrix[match_id], axis=0)
        self.Not_Matched_Detections = self.detections

    def chongdie_update(self, All_Robots):
        cost_matrix = np.zeros((self.robots_num, self.robots_num))
        for robots_id in range(len(All_Robots)):
            bbox = All_Robots[robots_id].xywh.T.astype(float)
            candidates = np.asarray([All_Robots[robots_id].xywh for i in range(self.robots_num)])
            candidates = np.squeeze(candidates)
            '''print("bbox", bbox)
            print("candidates", candidates)'''
            cost_matrix[robots_id, :] = 1. - self.get_iou(bbox, candidates)
        min_matrix = np.min(cost_matrix, axis=1)
        min_arg_matrix = np.argmin(cost_matrix, axis=1)
        # print("min_matraix", min_matrix)
        for match_id in range(len(min_arg_matrix)):
            # print("match_id", match_id)
            # print("min_matrix[match_id]", min_matrix[match_id])
            if min_matrix[match_id] < 0.3:
                print("begin to qingchu")
                if All_Robots[match_id].Armor_Confidence >= All_Robots[min_arg_matrix[match_id]].Armor_Confidence:
                    init_measurement = np.array([[0], [0], [0], [0]])
                    init_robot = robot_sort.Robot(init_measurement)
                    All_Robots[min_arg_matrix[match_id]] = init_robot
                else:
                    init_measurement = np.array([[0], [0], [0], [0]])
                    init_robot = robot_sort.Robot(init_measurement)
                    All_Robots[match_id] = init_robot
        return All_Robots

    def Not_Match_Update(self, All_Robots, frame):
        if len(self.detections) == 0:
            return All_Robots
        for Not_Matched_id in range(len(self.Not_Matched_Detections)):
            kuang = frame[int(self.detections[Not_Matched_id].tlwh[1]):int((
                    self.detections[Not_Matched_id].tlwh[1] + self.detections[Not_Matched_id].tlwh[3])),
                    int(self.detections[Not_Matched_id].tlwh[0]):int((
                            self.detections[Not_Matched_id].tlwh[0] + self.detections[Not_Matched_id].tlwh[2]))]
            kuang = cv2.resize(kuang, (300, 300))
            '''cv2.imshow("kuang", kuang)
            cv2.waitKey(0)'''
            result_armor = model_armor(kuang)
            for h in result_armor:
                armor_boxes = h.boxes
                armor_conf = armor_boxes.conf.cpu().numpy().astype(float)
                armor_id = armor_boxes.cls.cpu().numpy().astype(int)
                '''max_armor_conf = np.min(armor_conf, axis=0)
                max_armor_id = np.argmin(armor_conf, axis=0)'''

                for id in range(len(armor_conf)):
                    if 7 <= armor_id[id] < 14:
                        armor_id[id] = armor_id[id] - 7
                    if 14 <= armor_id[id] < 21:
                        armor_id[id] = armor_id[id] - 14
                    if 21 <= armor_id[id] < 28:
                        armor_id[id] = armor_id[id] - 21
                    # print("armor_id[id]", armor_id[id])
                    if armor_conf[id] > 0.5:
                        if armor_conf[id] > All_Robots[armor_id[id]].Armor_Confidence:
                            xyah = np.array([[self.Not_Matched_Detections[Not_Matched_id].to_xyah()[0]],
                                             [self.Not_Matched_Detections[Not_Matched_id].to_xyah()[1]],
                                             [self.Not_Matched_Detections[Not_Matched_id].to_xyah()[2]],
                                             [self.Not_Matched_Detections[Not_Matched_id].to_xyah()[3]]])
                            new_robot = robot_sort.Robot(xyah, armor_conf[id], armor_id[id])
                            for i in range(len(All_Robots)):
                                if xyah == All_Robots[i].mean[0:3]:
                                    if armor_conf[id] >= All_Robots[i].Armor_Confidence:
                                        init_measurement = np.array([[0], [0], [0], [0]])
                                        init_robot = robot_sort.Robot(init_measurement)
                                        All_Robots[i] = init_robot
                                        All_Robots[armor_id[id]] = new_robot
                                    else:
                                        init_measurement = np.array([[0], [0], [0], [0]])
                                        init_robot = robot_sort.Robot(init_measurement)
                                        All_Robots[armor_id[id]] = init_robot
                                else:
                                    All_Robots[armor_id[id]] = new_robot
        return All_Robots

    def iou_matrix(self):
        if len(self.detections) == 0:
            return
        cost_matrix = np.zeros((self.robots_num, self.detections_num))
        for robots_id in range(self.robots_num):
            '''print("robots", self.robots)
            print("robots_id", robots_id)'''
            bbox = self.robots[robots_id].xywh.T
            candidates = np.asarray([self.detections[i].tlwh for i in range(self.detections_num)])
            # print("iou_candidates", candidates)
            # print("bbox", bbox)
            # print("condidates", candidates)
            # print("get_iou", self.get_iou(bbox, candidates))
            cost_matrix[robots_id, :] = 1. - self.get_iou(bbox, candidates)
        return cost_matrix

    def get_iou(self, bbox, candidates):
        # bbox 即为 预测的robot的位置信息xywh
        # candidates 即为 所有这一时刻真实框的位置信息，用来与bbox匹配
        """Computer intersection over union.

        Parameters
        ----------
        bbox : ndarray
            A bounding box in format `(top left x, top left y, width, height)`.
        candidates : ndarray
            A matrix of candidate bounding boxes (one per row) in the same format
            as `bbox`.

        Returns
        -------
        ndarray
            The intersection over union in [0, 1] between the `bbox` and each
            candidate. A higher score means a larger fraction of the `bbox` is
            occluded by the candidate.

        """
        bbox_tl, bbox_br = bbox[0][:2], bbox[0][:2] + bbox[0][2:]
        if not candidates.any():
            # print("retrun not_candidates")
            return 0
        else:
            candidates_tl = candidates[:, :2]
            candidates_br = candidates[:, :2] + candidates[:, 2:]
            '''print("bbox", bbox)
            print("candidates_tl", candidates_br)
            print("candidates_br", candidates_tl[:, 1])
            print("bbox_tl", bbox_tl)
            print("bbox_br", bbox_br)'''
            tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
            np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
            br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
            np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
            wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        # print("area_intersection", area_intersection)
        area_bbox = bbox[2:].prod()
        # print("area_box", area_bbox)
        area_candidates = candidates[:, 2:].prod(axis=1)
        # print("area_candidates", area_candidates)
        return (area_intersection / (area_bbox + area_candidates - area_intersection)).T


if __name__ == "__main__":
    measurement1 = np.array([[230], [245], [1.2], [50]])
    measurement2 = np.array([[583], [830], [1.1], [60]])
    robot1 = robot_sort.Robot(measurement1)
    robot2 = robot_sort.Robot(measurement2)
    robot1.change_from_xyah_to_xywh()
    robot2.change_from_xyah_to_xywh()
    Robotor = robots.Robotor()
    Robotor.Robots.append(robot1)
    Robotor.Robots.append(robot2)
    tlwh1 = np.array([[210], [200], [60], [50]])
    tlwh2 = np.array([[600], [400], [60], [50]])
    tlwh3 = np.array([[540], [790], [84], [70]])
    tlwh4 = np.array([[0], [100], [60], [50]])
    detection1 = detection.Detection(tlwh1, 0.5, 0)
    detection2 = detection.Detection(tlwh2, 0.5, 0)
    detection3 = detection.Detection(tlwh3, 0.5, 0)
    detection4 = detection.Detection(tlwh4, 0.5, 0)
    detection_ = []
    detection_.append(detection1)
    detection_.append(detection2)
    detection_.append(detection3)
    detection_.append(detection4)
    matcher = Matcher(Robotor.Robots, detection_)
    matcher.iou_match()
