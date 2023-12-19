# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import new_kalman
from scipy.optimize import linear_sum_assignment
import detection
import robot_sort
import match


class Robotor:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, Robots, frame, metric=None, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = new_kalman.KalmanFilter()
        self.Robots = Robots
        self._next_id = 1
        self.frame = frame

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for Robot_id in range(len(self.Robots)):
            '''print("Robots", self.Robots)
            print("Robots[Robot_id]", self.Robots[Robot_id])
            print("mean",self.Robots[Robot_id].mean)'''
            self.Robots[Robot_id].predict()

    def update(self, detections):
        matcher = match.Matcher(self.Robots, detections)
        matcher.iou_match()
        self.Robots = matcher.robots
        for Robot_id in range(len(self.Robots)):
            if self.Robots[Robot_id].Armor_Confidence == 0:
                continue
            if self.Robots[Robot_id].Is_Have_Measurement is False:
                self.Robots[Robot_id].Not_Matched_Num += 1
                if self.Robots[Robot_id].Not_Matched_Num > 60:
                    init_measurement = np.array([[0], [0], [0], [0]])
                    init_robot = robot_sort.Robot(init_measurement)
                    self.Robots[Robot_id] = init_robot
            self.Robots[Robot_id].update(self.Robots[Robot_id].measurement)
            self.Robots[Robot_id].measurement = None
        All_Robots = matcher.Not_Match_Update(self.Robots, self.frame)
        All_Robots = matcher.chongdie_update(All_Robots)
        return All_Robots

    '''# 修改距离更新函数
    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)'''


    '''def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = .gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        measurement = np.array([mean[0]], [mean[1], [mean[2]], [mean[3]]])
        self.Robots.append(robot_sort.Robot(mean))
        self._next_id += 1'''
