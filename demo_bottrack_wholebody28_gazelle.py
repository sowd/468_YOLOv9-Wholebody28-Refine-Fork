#!/usr/bin/env python

"""
runtime: https://github.com/microsoft/onnxruntime

pip install onnxruntime or pip install onnxruntime-gpu
pip install lap==0.4.0 scipy==1.10.1 opencv-contrib-python==4.9.0.80
"""
from __future__ import annotations
import os
import re
import sys
import copy
import cv2
try:
    import onnx
    import onnxruntime # type: ignore
    from sne4onnx import extraction
    from sor4onnx import rename
except:
    pass
import time
from pprint import pprint
import lap
import requests # type: ignore
import subprocess
import numpy as np
import scipy.linalg
from enum import Enum
from collections import OrderedDict, deque
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentTypeError
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod
import json

# https://developer.nvidia.com/cuda-gpus
NVIDIA_GPU_MODELS_CC = [
    'RTX 3050', 'RTX 3060', 'RTX 3070', 'RTX 3080', 'RTX 3090',
]

ONNX_TRTENGINE_SETS = {
    'yolox_x_body_head_hand_face_0076_0.5228_post_1x3x480x640_score015_iou080_box050.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_14915622583698702352_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_14915622583698702352_1_1_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_14915622583698702352_1_1_fp16_sm86.profile',
    ],
    'face-reidentification-retail-0095_NMx3x128x128_post_feature_only.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_tf2onnx_2180071764421166639_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_tf2onnx_2180071764421166639_0_0_fp16_sm86.profile',
        'TensorrtExecutionProvider_TRTKernel_graph_tf2onnx_2180071764421166639_1_1_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_tf2onnx_2180071764421166639_1_1_fp16_sm86.profile',
    ],
    'mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_0_0_fp16_sm86.profile',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_1_1_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_1_1_fp16_sm86.profile',
    ],
}

BOX_COLORS = [
    [(216, 67, 21),"Front"],
    [(255, 87, 34),"Right-Front"],
    [(123, 31, 162),"Right-Side"],
    [(255, 193, 7),"Right-Back"],
    [(76, 175, 80),"Back"],
    [(33, 150, 243),"Left-Back"],
    [(156, 39, 176),"Left-Side"],
    [(0, 188, 212),"Left-Front"],
]

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

class Box(ABC):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, x1_norm: float = 0, y1_norm: float = 0, x2_norm: float = 0, y2_norm: float = 0, generation: int = -1, gender: int = -1, handedness: int = -1, head_pose: int = -1, is_used: bool = False):
        self.trackid: int = trackid
        self.classid: int = classid
        self.score: float = score
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.cx: int = cx
        self.cy: int = cy
        self.x1_norm: float = x1_norm
        self.y1_norm: float = y1_norm
        self.x2_norm: float = x2_norm
        self.y2_norm: float = y2_norm
        self.generation: int = generation
        self.gender: int = gender
        self.handedness: int = handedness
        self.head_pose: int = head_pose
        self.is_used: bool = is_used

class Body(Box):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool, box: Box, head: Box, hand1: Box, hand2: Box):
        super().__init__(trackid=trackid, classid=classid, score=score, x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, is_used=is_used)
        self.box: Box = box
        self.head: Head = head
        self.hand1: Hand = hand1
        self.hand2: Hand = hand2

class Head(Box):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool, box: Box, face: Box, face_landmarks: np.ndarray):
        super().__init__(trackid=trackid, classid=classid, score=score, x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, is_used=is_used)
        self.box: Box = box
        self.face: Box = face
        self.face_landmarks: np.ndarray = face_landmarks

class Face(Box):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool, box: Box):
        super().__init__(trackid=trackid, classid=classid, score=score, x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, is_used=is_used)
        self.box: Box = box

class Hand(Box):
    def __init__(self, trackid: int, classid: int, score: float, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, is_used: bool, box: Box):
        super().__init__(trackid=trackid, classid=classid, score=score, x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, is_used=is_used)
        self.box: Box = box

class Gaze:
    def __init__(self, trackid: int, head_x: int, head_y: int, target_x: int, target_y: int):
        self.trackid: int = trackid
        self.head_x: int = head_x
        self.head_y: int = head_y
        self.target_x: int = target_x
        self.target_y: int = target_y

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh

    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is taken as direct observation of the state space (linear
    observation model).
    """

    """
    Table for the 0.95 quantile of the chi-square distribution with N degrees of
    freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
    function and used as Mahalanobis gating threshold.
    """
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919
    }

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean: np.ndarray = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, w the width, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain: np.ndarray = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean: np.ndarray = mean + np.dot(innovation, kalman_gain.T)
        new_covariance: np.ndarray = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4

class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    body_curr_feature = None
    face_curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, feature_history: int, body: Body, body_feature: np.ndarray=None, face_feature: np.ndarray=None):
        """STrack

        Parameters
        ----------
        tlwh: np.ndarray
            Top-left, width, height. [x1, y1, w, h]

        score: float
            Object detection score.

        feature_history: int
            Number of features to be retained in history.

        body: Body

        body_feature: Optional[np.ndarray]
            Features obtained from the feature extractor.

        face_feature: Optional[np.ndarray]
            Features obtained from the feature extractor.
        """
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: KalmanFilter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.alpha = 0.9
        self.feature_history = feature_history

        self.body = body

        # Body features
        self.body_smooth_feature = None
        self.body_curr_feature = None
        self.body_features = deque([], maxlen=feature_history)
        if body_feature is not None:
            self.update_body_features(body_feature)

        # Face features
        self.face_smooth_feature = None
        self.face_curr_feature = None
        self.face_features = deque([], maxlen=feature_history)
        if face_feature is not None:
            self.update_face_features(face_feature)

    def update_body_features(self, feature: np.ndarray):
        # Skip processing because it has already been
        # normalized in the post-processing process of ONNX.
        # feature /= np.linalg.norm(feature)
        self.body_curr_feature = feature
        if self.body_smooth_feature is None:
            self.body_smooth_feature = feature
        else:
            self.body_smooth_feature = self.alpha * self.body_smooth_feature + (1 - self.alpha) * feature
        self.body_features.append(feature)
        self.body_smooth_feature /= np.linalg.norm(self.body_smooth_feature)

    def update_face_features(self, feature: np.ndarray):
        # Skip processing because it has already been
        # normalized in the post-processing process of ONNX.
        # feature /= np.linalg.norm(feature)
        self.face_curr_feature = feature
        if self.face_smooth_feature is None:
            self.face_smooth_feature = feature
        else:
            self.face_smooth_feature = self.alpha * self.face_smooth_feature + (1 - self.alpha) * feature
        self.face_features.append(feature)
        self.face_smooth_feature /= np.linalg.norm(self.face_smooth_feature)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: List[STrack]):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: List[STrack], H: np.ndarray=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.body_curr_feature is not None:
            self.update_body_features(new_track.body_curr_feature)
        if new_track.face_curr_feature is not None:
            self.update_face_features(new_track.face_curr_feature)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.body = new_track.body

    def update(self, new_track: STrack, frame_id: int):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.body_curr_feature is not None:
            self.update_body_features(new_track.body_curr_feature)
        if new_track.face_curr_feature is not None:
            self.update_face_features(new_track.face_curr_feature)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.body = new_track.body

    def propagate_trackid_to_related_objects(self):
        if self.body is not None:
            self.body.trackid = self.track_id
            self.body.box.trackid = self.track_id
            if self.body.head is not None:
                self.body.head.trackid = self.track_id
                self.body.head.box.trackid = self.track_id
                if self.body.head.face is not None:
                    self.body.head.face.trackid = self.track_id
                    self.body.head.face.box.trackid = self.track_id
            if self.body.hand1 is not None:
                self.body.hand1.trackid = self.track_id
                self.body.hand1.box.trackid = self.track_id
            if self.body.hand2 is not None:
                self.body.hand2.trackid = self.track_id
                self.body.hand2.box.trackid = self.track_id

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70

    _input_shapes: List[List[int | str]] = []
    _input_names: List[str] = []
    _input_dtypes: List[np.dtype] = []
    _output_shapes: List[List[int | str]] = []
    _output_names: List[str] = []

    _input_shapes_postprocess: List[List[int | str]] = []
    _input_names_postprocess: List[str] = []
    _input_dtypes_postprocess: List[np.dtype] = []
    _output_shapes_postprocess: List[List[int | str]] = []
    _output_names_postprocess: List[str] = []

    _mean: np.ndarray = np.array([0.000, 0.000, 0.000], dtype=np.float32)
    _std: np.ndarray = np.array([1.000, 1.000, 1.000], dtype=np.float32)

    # onnx/tflite
    _interpreter = None
    _interpreter_postprocess = None
    _providers = None
    _swap: Tuple = (2, 0, 1)
    _h_index: int = 2
    _w_index: int = 3
    _norm_shape: List = [1,3,1,1]
    _class_score_th: float

    # onnx
    _onnx_dtypes_to_np_dtypes: Dict[str, np.dtype] = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        model_path_post: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        mean: Optional[np.ndarray] = np.array([0.000, 0.000, 0.000], dtype=np.float32),
        std: Optional[np.ndarray] = np.array([1.000, 1.000, 1.000], dtype=np.float32),
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._model_path_post = model_path_post
        self._model = None
        self._model_postprocess = None
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3

            # Initialize model body
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

            onnx_graph: onnx.ModelProto = onnx.load(model_path)
            if onnx_graph.graph.node[0].op_type == "Resize":
                first_resize_op: List[onnx.ValueInfoProto] = [i for i in onnx_graph.graph.value_info if i.name == "prep/Resize_output_0"]
                if first_resize_op:
                    self._input_shapes = [[d.dim_value for d in first_resize_op[0].type.tensor_type.shape.dim]]
                else:
                    self._input_shapes = [
                        input.shape for input in self._interpreter.get_inputs()
                    ]
            else:
                self._input_shapes = [
                    input.shape for input in self._interpreter.get_inputs()
                ]


            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3
            self._norm_shape = [1,3,1,1]

            # Support for TensorRT 9.x+
            # Initialize model post-process
            if any((p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider" for p in providers) and model_path_post:
                self._interpreter_postprocess = \
                    onnxruntime.InferenceSession(
                        model_path_post,
                        sess_options=session_option,
                        providers=['CPUExecutionProvider'],
                    )
                self._input_names_postprocess = [
                    input.name for input in self._interpreter_postprocess.get_inputs()
                ]
                self._input_dtypes_postprocess = [
                    self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter_postprocess.get_inputs()
                ]
                self._output_shapes_postprocess = [
                    output.shape for output in self._interpreter_postprocess.get_outputs()
                ]
                self._output_names_postprocess = [
                    output.name for output in self._interpreter_postprocess.get_outputs()
                ]
                self._model_postprocess = self._interpreter_postprocess.run

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2
            self._norm_shape = [1,1,1,3]

        self._mean = mean.reshape(self._norm_shape)
        self._std = std.reshape(self._norm_shape)
        self._class_score_th = obj_class_score_th

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            # Support for TensorRT 9.x+
            # Isolation of NMS
            if self._model_postprocess:
                datas = {
                    f'{input_name}': input_data \
                        for input_name, input_data in zip(self._output_names, outputs)
                }
                outputs = [
                    output for output in \
                        self._model_postprocess(
                            output_names=self._output_names_postprocess,
                            input_feed=datas,
                        )
                ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

    # Support for TensorRT 9.x+
    def model_split(
        self,
        *,
        model_path: str,
        output_model_path: str,
        runtime: Optional[str] = 'onnx',
        input_op_names: List[str] = ['input'],
        output_op_names: List[str] = ['output'],
    ) -> onnx.ModelProto:
        """https://github.com/PINTO0309/sne4onnx

        Parameters
        ----------
        model_path: str
            ONNX file path for YOLOv9

        output_model_path: str
            ONNX file path for YOLOv9

        runtime: Optional[str]
            Default: 'onnx'

        input_op_names: List[str]
            Default: ['input']

        output_op_names: List[str]
            Default: ['output']

        Returns
        -------
        extracted_model: onnx.ModelProto
        """
        if runtime != 'onnx':
            raise NotImplementedError()
        extracted_model = extraction(
            input_op_names=input_op_names,
            output_op_names=output_op_names,
            input_onnx_file_path=model_path,
            output_onnx_file_path=output_model_path,
            non_verbose=True,
        )
        return extracted_model

    # Support for TensorRT 9.x+
    def model_op_rename(
        self,
        *,
        model_path: str,
        output_model_path: str,
        runtime: Optional[str] = 'onnx',
        old_new: List[str] = ['input', 'input'],
        mode: str = 'full',
        search_mode: str = 'exact_match',
    ) -> onnx.ModelProto:
        """https://github.com/PINTO0309/sor4onnx

        Parameters
        ----------
        model_path: str
            ONNX file path for YOLOv9

        output_model_path: str
            ONNX file path for YOLOv9

        runtime: Optional[str]
            Default: 'onnx'

        old_new: List[str]
            Default: ['input', 'input']

        mode: str
            Default: 'full'

        search_mode: str
            Default: 'exact_match'

        Returns
        -------
        renamed_model: onnx.ModelProto
        """
        if runtime != 'onnx':
            raise NotImplementedError()
        renamed_model = rename(
            old_new=old_new,
            input_onnx_file_path=model_path,
            output_onnx_file_path=output_model_path,
            non_verbose=True,
        )
        return renamed_model

class YOLOv9(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolov9_e_wholebody28_refine_post_0100_1x3x480x640.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOv9. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOv9

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        keypoint_th: Optional[float]
            Keypoints score threshold. Default: 0.25

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        # Support for TensorRT 9.x+
        splited_model_body_path = model_path
        splited_model_post_path = ''
        if runtime == 'onnx' and any((p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider" for p in providers):
            # Support for TensorRT 9.x+, Isolation of NMS

            # Model body part generation
            splited_model_body_path = f"{os.path.splitext(os.path.basename(model_path))[0]}_body.onnx"
            if not os.path.isfile(splited_model_body_path):
                super().model_split(
                    model_path=model_path,
                    output_model_path=splited_model_body_path,
                    runtime=runtime,
                    input_op_names=['input_bgr'],
                    output_op_names=['x1y1x2y2', 'main01_y1x1y2x2', 'main01_scores'],
                )
                super().model_op_rename(
                    model_path=splited_model_body_path,
                    output_model_path=splited_model_body_path,
                    runtime=runtime,
                    old_new=['main01_scores', 'scores'],
                    mode='outputs',
                    search_mode='exact_match',
                )
                super().model_op_rename(
                    model_path=splited_model_body_path,
                    output_model_path=splited_model_body_path,
                    runtime=runtime,
                    old_new=['main01_y1x1y2x2', 'y1x1y2x2'],
                    mode='outputs',
                    search_mode='exact_match',
                )

            # Model post-process part generation
            splited_model_post_path = f"{os.path.splitext(os.path.basename(model_path))[0]}_post.onnx"
            if not os.path.isfile(splited_model_post_path):
                super().model_split(
                    model_path=model_path,
                    output_model_path=splited_model_post_path,
                    runtime=runtime,
                    input_op_names=['x1y1x2y2', 'main01_y1x1y2x2', 'main01_scores'],
                    output_op_names=['batchno_classid_score_x1y1x2y2'],
                )
                super().model_op_rename(
                    model_path=splited_model_post_path,
                    output_model_path=splited_model_post_path,
                    runtime=runtime,
                    old_new=['main01_scores', 'scores'],
                    mode='inputs',
                    search_mode='exact_match',
                )
                super().model_op_rename(
                    model_path=splited_model_post_path,
                    output_model_path=splited_model_post_path,
                    runtime=runtime,
                    old_new=['main01_y1x1y2x2', 'y1x1y2x2'],
                    mode='inputs',
                    search_mode='exact_match',
                )

        super().__init__(
            runtime=runtime,
            model_path=splited_model_body_path,
            model_path_post=splited_model_post_path if splited_model_post_path else '',
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            keypoint_th=keypoint_th,
            providers=providers,
        )

        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, atrributes, is_used=False]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]

        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )

        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]. [instances, [batchno, classid, score, x1, y1, x2, y2]].

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        box_score_threshold: float = min([self._obj_class_score_th, self._attr_class_score_th, self._keypoint_th])

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > box_score_threshold
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                # Object filter
                for box, score in zip(boxes_keep, scores_keep):
                    classid = int(box[1])
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    x1_norm = max(0, box[3]) / self._input_shapes[0][self._w_index]
                    y1_norm = max(0, box[4]) / self._input_shapes[0][self._h_index]
                    x2_norm = min(box[5], self._input_shapes[0][self._w_index]) / self._input_shapes[0][self._w_index]
                    y2_norm = min(box[6], self._input_shapes[0][self._h_index]) / self._input_shapes[0][self._h_index]
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    result_boxes.append(
                        Box(
                            trackid=0,
                            classid=classid,
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            x1_norm=x1_norm,
                            y1_norm=y1_norm,
                            x2_norm=x2_norm,
                            y2_norm=y2_norm,
                            cx=cx,
                            cy=cy,
                            generation=-1, # -1: Unknown, 0: Adult, 1: Child
                            gender=-1, # -1: Unknown, 0: Male, 1: Female
                            handedness=-1, # -1: Unknown, 0: Left, 1: Right
                            head_pose=-1, # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
                        )
                    )
                # Object filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [0,5,6,7,16,17,18,19,20,23,24,25,27] and box.score >= self._obj_class_score_th) or box.classid not in [0,5,6,7,16,17,18,19,20,23,24,25,27]
                ]
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [1,2,3,4,8,9,10,11,12,13,14,15] and box.score >= self._attr_class_score_th) or box.classid not in [1,2,3,4,8,9,10,11,12,13,14,15]
                ]
                # Keypoint filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [21,22,26] and box.score >= self._keypoint_th) or box.classid not in [21,22,26]
                ]

                # Adult, Child merge
                # classid: 0 -> Body
                #   classid: 1 -> Adult
                #   classid: 2 -> Child
                # 1. Calculate Adult and Child IoUs for Body detection results
                # 2. Connect either the Adult or the Child with the highest score and the highest IoU with the Body.
                # 3. Exclude Adult and Child from detection results
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]
                # Male, Female merge
                # classid: 0 -> Body
                #   classid: 3 -> Male
                #   classid: 4 -> Female
                # 1. Calculate Male and Female IoUs for Body detection results
                # 2. Connect either the Male or the Female with the highest score and the highest IoU with the Body.
                # 3. Exclude Male and Female from detection results
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]
                # HeadPose merge
                # classid: 7 -> Head
                #   classid:  8 -> Front
                #   classid:  9 -> Right-Front
                #   classid: 10 -> Right-Side
                #   classid: 11 -> Right-Back
                #   classid: 12 -> Back
                #   classid: 13 -> Left-Back
                #   classid: 14 -> Left-Side
                #   classid: 15 -> Left-Front
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8,9,10,11,12,13,14,15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8,9,10,11,12,13,14,15]]
                # Left and right hand merge
                # classid: 23 -> Hand
                #   classid: 24 -> Left-Hand
                #   classid: 25 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 23]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [24, 25]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [24, 25]]
        return result_boxes

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 24:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 25:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

class GazeLLE(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'gazelle_dinov2_vitb14_inout_1x3x448x448_1xNx4.onnx',
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for GazeLLE. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for GazeLLE

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in GazeLLE
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in GazeLLE

    def __call__(
        self,
        image: np.ndarray,
        head_boxes: List[Box],
        disable_attention_heatmap_mode: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image, BGR

        head_boxes: List[Box]
            Head boxes

        disable_attention_heatmap_mode: bool

        Returns
        -------
        result_image: np.ndarray
            BGR, uint8[image_height, image_width, 3]

        heatmaps: np.ndarray
            [1, 64, 64]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        head_boxes_xyxy = []
        for head_box in head_boxes:
            head_boxes_xyxy.append([head_box.x1_norm, head_box.y1_norm, head_box.x2_norm, head_box.y2_norm])
        inferecne_head_boxes = np.asarray([head_boxes_xyxy], dtype=self._input_dtypes[1])
        outputs = super().__call__(input_datas=[inferece_image, inferecne_head_boxes])
        heatmaps = outputs[0]
        if len(outputs) == 2:
            inout = outputs[1]
        # PostProcess
        result_image, resized_heatmatps = \
            self._postprocess(
                image_bgr=temp_image,
                heatmaps=heatmaps,
                disable_attention_heatmap_mode=disable_attention_heatmap_mode,
            )
        return result_image, resized_heatmatps

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = cv2.resize(image, (448, 448))
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image_bgr: np.ndarray,
        heatmaps: np.ndarray,
        disable_attention_heatmap_mode: bool,
    ) -> np.ndarray:
        """_postprocess

        Parameters
        ----------
        image_bgr: np.ndarray
            Entire image.

        heatmaps: np.ndarray
            float32[heads, 64, 64]

        disable_attention_heatmap_mode: bool

        Returns
        -------
        result_image: uint8[image_height, image_width, 3]
            BGR

        resized_heatmatps: uint8[image_height, image_width]
            Single-channel
        """
        image_height = image_bgr.shape[0]
        image_width = image_bgr.shape[1]
        if not disable_attention_heatmap_mode:
            image_rgb = image_bgr[..., ::-1]
            heatmaps_all: np.ndarray = np.sum(heatmaps, axis=0) # [64, 64]
            heatmaps_all = heatmaps_all * 255
            heatmaps_all = heatmaps_all.astype(np.uint8)
            heatmaps_all = Image.fromarray(heatmaps_all).resize((image_width, image_height), Image.Resampling.BILINEAR)
            heatmaps_all = plt.cm.jet(np.array(heatmaps_all) / 255.0)
            heatmaps_all = (heatmaps_all[:, :, :3] * 255).astype(np.uint8)
            heatmaps_all = Image.fromarray(heatmaps_all).convert("RGBA")
            heatmaps_all.putalpha(128)
            image_rgba = Image.alpha_composite(Image.fromarray(image_rgb).convert("RGBA"), heatmaps_all)
            image_bgr = cv2.cvtColor(np.asarray(image_rgba)[..., [2,1,0,3]], cv2.COLOR_BGRA2BGR)
        else:
            pass

        heatmap_list = [cv2.resize(heatmap[..., None], (image_width, image_height)) for heatmap in heatmaps]
        resized_heatmatps = np.asarray(heatmap_list)

        return image_bgr, resized_heatmatps

class FastReID(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx',
        providers: Optional[List] = None,
    ):
        """FastReID

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for FastReID. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for FastReID

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )
        self.feature_size = self._output_shapes[1][-1]

    def __call__(
        self,
        *,
        base_images: List[np.ndarray],
        target_features: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FastReID

        Parameters
        ----------
        base_image: List[np.ndarray]
            Object images [N, 3, H, W]

        target_features: List[np.ndarray]
            features [M, 2048]

        Returns
        -------
        similarities: np.ndarray
            features [N, M]

        base_features: np.ndarray
            features [M, 2048]
        """
        temp_base_images = copy.deepcopy(base_images)
        temp_target_features = copy.deepcopy(target_features)

        # PreProcess
        temp_base_images = \
            self._preprocess(
                base_images=temp_base_images,
            )

        # Inference
        outputs = super().__call__(input_datas=[temp_base_images, temp_target_features])
        similarities = outputs[0]
        base_features = outputs[1]
        return similarities, base_features

    def _preprocess(
        self,
        *,
        base_images: List[np.ndarray],
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        base_images: List[np.ndarray]
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        stacked_images_N: np.ndarray
            Resized and normalized image. [N, 3, H, W]
        """
        # Resize + Transpose
        resized_base_images_np: np.ndarray = None
        resized_base_images_list: List[np.ndarray] = []
        for base_image in base_images:
            resized_base_image: np.ndarray = \
                cv2.resize(
                    src=base_image,
                    dsize=(
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
            resized_base_image = resized_base_image[..., ::-1] # BGR to RGB
            resized_base_image = resized_base_image.transpose(self._swap)
            resized_base_images_list.append(resized_base_image)
        resized_base_images_np = np.asarray(resized_base_images_list)
        resized_base_images_np = (resized_base_images_np / 255.0 - self._mean) / self._std
        resized_base_images_np = resized_base_images_np.astype(self._input_dtypes[0])
        return resized_base_images_np

    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class FaceReidentificationRetail0095(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'face-reidentification-retail-0095_NMx3x128x128_post_feature_only.onnx',
        providers: Optional[List] = None,
    ):
        """FaceReidentificationRetail0095

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for FaceReidentificationRetail0095. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for FaceReidentificationRetail0095

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self.feature_size = self._output_shapes[0][-1]

    def __call__(
        self,
        *,
        base_images: List[np.ndarray],
        target_features: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FaceReidentificationRetail0095

        Parameters
        ----------
        base_image: List[np.ndarray]
            Object images [N, 3, H, W]

        target_features: List[np.ndarray]
            features [M, 256]

        Returns
        -------
        similarities: np.ndarray
            features [N, M]

        base_features: np.ndarray
            features [M, 256]
        """
        temp_base_images = copy.deepcopy(base_images)
        temp_target_features = copy.deepcopy(target_features)

        # PreProcess
        temp_base_images = \
            self._preprocess(
                base_images=temp_base_images,
            )

        # Inference
        outputs = super().__call__(input_datas=[temp_base_images, temp_target_features])
        similarities = outputs[0]
        base_features = outputs[1]
        return similarities, base_features

    def _preprocess(
        self,
        *,
        base_images: List[np.ndarray],
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        base_images: List[np.ndarray]
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        stacked_images_N: np.ndarray
            Resized and normalized image. [N, 3, H, W]
        """
        # Resize + Transpose
        resized_base_images_np: np.ndarray = None
        resized_base_images_list: List[np.ndarray] = []
        for base_image in base_images:
            resized_base_image: np.ndarray = \
                cv2.resize(
                    src=base_image,
                    dsize=(
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
            resized_base_image = resized_base_image.transpose(self._swap)
            resized_base_images_list.append(resized_base_image)
        resized_base_images_np = np.asarray(resized_base_images_list)
        resized_base_images_np = resized_base_images_np.astype(self._input_dtypes[0])
        return resized_base_images_np

    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class BoTSORT(object):
    def __init__(
        self,
        object_detection_model,
        body_feature_extractor_model,
        face_feature_extractor_model,
        frame_rate: int=30,
    ):

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh: float = 0.40 # tracking confidence threshold Default: 0.6
        self.track_low_thresh: float = 0.1 # lowest detection threshold valid for tracks Default: 0.1
        self.new_track_thresh: float = 0.9 # new track thresh Default: 0.7
        self.match_thresh: float = 0.8 # matching threshold for tracking Default: 0.8
        self.track_buffer: int = 300 # the frames for keep lost tracks Default: 30
        self.feature_history: int = 300 # the frames for keep features Default: 50
        self.proximity_thresh: float = 0.5 # threshold for rejecting low overlap reid matches Default: 0.5
        self.appearance_thresh: float = 0.25 # threshold for rejecting low appearance similarity reid matches Default: 0.25
        self.buffer_size: int = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost: int = self.buffer_size
        self.kalman_filter: KalmanFilter = KalmanFilter()

        # Object detection module
        self.detector: YOLOv9 = object_detection_model

        # BodyReID module
        self.body_encoder: FastReID = body_feature_extractor_model
        self.body_strack_features: List[np.ndarray] = []

        # FaceReID module
        self.face_encoder: FaceReidentificationRetail0095 = face_feature_extractor_model
        self.face_strack_features: List[np.ndarray] = []

    def update(self, image: np.ndarray, detected_boxes: List[Box]) -> List[STrack]:
        self.frame_id += 1
        activated_starcks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        debug_image = copy.deepcopy(image)

        # Generate Body object
        body_boxes = [
            Body(
                trackid=0,
                classid=box.classid,
                score=box.score,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                cx=box.cx,
                cy=box.cy,
                box=box,
                head=None,
                hand1=None,
                hand2=None,
                is_used=False,
            ) for box in detected_boxes if box.classid == 0 # Body
        ]

        # Generate Head object
        head_boxes: List[Head] = [
            Head(
                trackid=0,
                classid=box.classid,
                score=box.score,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                cx=box.cx,
                cy=box.cy,
                box=box,
                face=None,
                face_landmarks=None,
                is_used=False,
            ) for box in detected_boxes if box.classid == 7 # Head
        ]

        # Generate Hand object
        hand_boxes: List[Hand] = [
            Hand(
                trackid=0,
                classid=box.classid,
                score=box.score,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                cx=box.cx,
                cy=box.cy,
                is_used=False,
                box=box,
            ) for box in detected_boxes if box.classid == 23 # Hand
        ]

        # Generate Face object
        face_boxes: List[Face] = [
            Face(
                trackid=0,
                classid=box.classid,
                score=box.score,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                cx=box.cx,
                cy=box.cy,
                is_used=False,
                box=box,
            ) for box in detected_boxes if box.classid == 16 # Face
        ]

        # Associate Face object to Head object
        if len(face_boxes) > 0:
            for head_box in head_boxes:
                closest_face_box: Box = \
                    find_most_relevant_object(
                        base_obj=head_box,
                        target_objs=face_boxes,
                    )
                if closest_face_box is not None:
                    head_box.face = closest_face_box

        # Associate Head_Face object to Body object
        if len(head_boxes) > 0:
            for body_box in body_boxes:
                closest_head_box: Box = \
                    find_most_relevant_object(
                        base_obj=body_box,
                        target_objs=head_boxes,
                    )
                if closest_head_box is not None:
                    body_box.head = closest_head_box

        # Associate Hand object to Body object
        if len(hand_boxes) > 0:
            for body_box in body_boxes:
                closest_hand_box1: Box = \
                    find_most_relevant_object(
                        base_obj=body_box,
                        target_objs=hand_boxes,
                    )
                if closest_hand_box1 is not None:
                    body_box.hand1 = closest_hand_box1

                closest_hand_box2: Box = \
                    find_most_relevant_object(
                        base_obj=body_box,
                        target_objs=hand_boxes,
                    )
                if closest_hand_box2 is not None:
                    body_box.hand2 = closest_hand_box2
        # Object detection =========================================================

        # Add newly detected tracklets to tracked_stracks
        unconfirmed_stracks: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Extract embeddings
        # Feature extraction of detected objects and
        # calculation of similarity between the extracted features and the previous feature.
        # At the first calculation, the previous features are treated as all zeros.
        # similarities: [N, M]
        # current_features: [N, 2048]
        person_images: List[np.ndarray] = [
            debug_image[box.y1:box.y2, box.x1:box.x2, :] for box in body_boxes
        ]
        face_images: List[np.ndarray] = [
            debug_image[body_box.head.face.y1:body_box.head.face.y2, body_box.head.face.x1:body_box.head.face.x2, :] \
                if body_box.head is not None and body_box.head.face is not None \
                    else np.zeros([d if isinstance(d, int) else 1 for d in self.face_encoder._input_shapes[0][1:]], dtype=np.float32).transpose(1,2,0) for body_box in body_boxes
        ]

        current_stracks: List[STrack] = []

        # Body feature extraction
        body_strack_features: List[np.ndarray] = []
        body_similarities: np.ndarray = None
        body_current_features: np.ndarray = None
        body_strack_features = [
            strack.body_curr_feature for strack in strack_pool
        ] if len(strack_pool) > 0 else np.zeros([0, self.body_encoder.feature_size], dtype=np.float32)
        if len(person_images) > 0:
            body_similarities_and_current_features: Tuple[np.ndarray, np.ndarray] = \
                self.body_encoder(
                    base_images=person_images,
                    target_features=body_strack_features,
                )
            body_similarities = body_similarities_and_current_features[0]
            body_similarities = body_similarities.transpose(1, 0) # N: boxes M: stracks, [N, M] -> [M, N]
            body_current_features = body_similarities_and_current_features[1]
        else:
            body_similarities = np.zeros([0, len(strack_pool)], dtype=np.float32).transpose(1, 0)
            body_current_features = np.zeros([0, self.body_encoder.feature_size], dtype=np.float32)

        # Face feature extraction
        face_strack_features: List[np.ndarray] = []
        face_similarities: np.ndarray = None
        face_current_features: np.ndarray = None
        face_strack_features = [
            strack.face_curr_feature for strack in strack_pool
        ] if len(strack_pool) > 0 else np.zeros([0, self.face_encoder.feature_size], dtype=np.float32)
        if len(face_images) > 0:
            face_similarities_and_current_features: Tuple[np.ndarray, np.ndarray] = \
                self.face_encoder(
                    base_images=face_images,
                    target_features=face_strack_features,
                )
            face_similarities = face_similarities_and_current_features[1]
            face_similarities = face_similarities.transpose(1, 0) # N: boxes M: stracks, [N, M] -> [M, N]
            face_current_features = face_similarities_and_current_features[0]
            # Workaround for problems with a series of abnormal values of 0.99999999
            close_to_value_mask = np.isclose(face_similarities, 0.9999999, atol=1e-08, rtol=1e-08)
            face_similarities[close_to_value_mask] = 0.0
        else:
            face_similarities = np.zeros([len(person_images), len(strack_pool)], dtype=np.float32).transpose(1, 0)
            face_current_features = np.zeros([len(person_images), self.face_encoder.feature_size], dtype=np.float32)

        current_stracks: List[STrack] = []
        body_current_similarities: np.ndarray = copy.deepcopy(body_similarities)
        face_current_similarities: np.ndarray = copy.deepcopy(face_similarities)
        low_score_current_stracks: List[STrack] = []
        if len(body_boxes) > 0:
            current_stracks: List[STrack] = [
                STrack(
                    tlwh=STrack.tlbr_to_tlwh(np.asarray([body.x1, body.y1, body.x2, body.y2])),
                    score=body.score,
                    body=body,
                    body_feature=body_current_feature,
                    face_feature=face_current_feature,
                    feature_history=self.feature_history,
                ) for body, body_current_feature, face_current_feature in zip(body_boxes, body_current_features, face_current_features) if body.score > self.track_high_thresh
            ]
            if len(body_boxes) != len(current_stracks) and len(current_stracks) > 0 and len(body_current_similarities) > 0:
                # body
                body_current_similarities = body_current_similarities.transpose(1, 0) # M: stracks N: boxes, [M, N] -> [N, M]
                body_current_similarities = np.asarray([
                    current_similarity for body, current_similarity in zip(body_boxes, body_current_similarities) if body.score > self.track_high_thresh
                ], dtype=np.float32)
                body_current_similarities = body_current_similarities.transpose(1, 0) # N: boxes M: stracks, [N, M] -> [M, N]
                # face
                face_current_similarities = face_current_similarities.transpose(1, 0) # M: stracks N: boxes, [M, N] -> [N, M]
                face_current_similarities = np.asarray([
                    current_similarity for body, current_similarity in zip(body_boxes, face_current_similarities) if body.score > self.track_high_thresh
                ], dtype=np.float32)
                face_current_similarities = face_current_similarities.transpose(1, 0) # N: boxes M: stracks, [N, M] -> [M, N]
            elif len(current_stracks) == 0 and len(body_current_similarities) > 0:
                pass
            elif len(current_stracks) > 0 and len(body_current_similarities) == 0:
                # body
                body_current_similarities = np.zeros([0, len(current_stracks)], dtype=np.float32)
                # face
                face_current_similarities = np.zeros([0, len(current_stracks)], dtype=np.float32)
            low_score_current_stracks: List[STrack] = [
                STrack(
                    tlwh=STrack.tlbr_to_tlwh(np.asarray([body.x1, body.y1, body.x2, body.y2])),
                    score=body.score,
                    body=body,
                    body_feature=body_current_feature,
                    face_feature=face_current_feature,
                    feature_history=self.feature_history
                ) for body, body_current_feature, face_current_feature in zip(body_boxes, body_current_features, face_current_features) if body.score <= self.track_high_thresh and body.score >= self.track_low_thresh
            ]

        # Calibration by camera motion is not performed.
        # STrack.multi_gmc(strack_pool, np.eye(2, 3, dtype=np.float32))
        # STrack.multi_gmc(unconfirmed, np.eye(2, 3, dtype=np.float32))

        # First association, with high score detection boxes
        ious_dists = iou_distance(strack_pool, current_stracks)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # @@@@@@@@@ Body + Face ReID
        emb_dists = 1.0 - body_current_similarities
        face_emb_dists = 1.0 - face_current_similarities
        emb_dists_comp = np.minimum(emb_dists, face_emb_dists)
        emb_dists_mask = emb_dists_comp > self.appearance_thresh
        emb_dists[emb_dists_mask] = 1.0
        # Improved stability when returning from out-of-view angle.
        # if the COS distance is smaller than the default value,
        # the IoU distance judgment result is ignored and priority
        # is given to the COS distance judgment result.
        ious_dists_mask = np.logical_and(emb_dists_mask, ious_dists_mask)
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track: STrack = strack_pool[itracked]
            det: STrack = current_stracks[idet]
            if track.state == TrackState.Tracked:
                track.update(current_stracks[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(new_track=det, frame_id=self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Second association, with low score detection boxes
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, low_score_current_stracks)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det: STrack = low_score_current_stracks[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(new_track=det, frame_id=self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        unconfirmed_boxes = [current_stracks[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed_stracks, unconfirmed_boxes)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        unconfirmed_strack_curr_features = \
            np.asarray([unconfirmed_strack.body_curr_feature for unconfirmed_strack in unconfirmed_stracks], dtype=np.float32) \
                if len(unconfirmed_stracks) > 0 else np.zeros([0, self.body_encoder.feature_size], dtype=np.float32)
        unconfirmed_boxes_features = \
            np.asarray([unconfirmed_box.body_curr_feature for unconfirmed_box in unconfirmed_boxes], dtype=np.float32) \
                if len(unconfirmed_boxes) > 0 else np.zeros([0, self.body_encoder.feature_size], dtype=np.float32)
        emb_dists = 1.0 - np.maximum(0.0, np.matmul(unconfirmed_strack_curr_features, unconfirmed_boxes_features.transpose(1, 0)))
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed_track: STrack = unconfirmed_stracks[itracked]
            unconfirmed_track.update(unconfirmed_boxes[idet], self.frame_id)
            activated_starcks.append(unconfirmed_track)
        for it in u_unconfirmed:
            track = unconfirmed_stracks[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Init new stracks
        for inew in u_detection:
            track = unconfirmed_boxes[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Merge
        self.tracked_stracks: List[STrack] = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks: List[STrack] = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks: List[STrack] = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks: List[STrack] = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        _ = [tracked_strack.propagate_trackid_to_related_objects() for tracked_strack in self.tracked_stracks]
        return self.tracked_stracks


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]):
    exists: Dict[int, int] = {}
    res: List[STrack] = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista: List[STrack], tlistb: List[STrack]):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack]):
    pdist: np.ndarray = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        strackp: STrack =stracksa[p]
        timep = strackp.frame_id - strackp.start_frame
        strackq: STrack =stracksb[q]
        timeq = strackq.frame_id - strackq.start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def bbox_iou(atlbr: np.ndarray, btlbr: np.ndarray) -> float:
    # atlbr: [x1, y1, x2, y2]
    # btlbr: [x1, y1, x2, y2]

    # Calculate areas of overlap
    inter_xmin = max(atlbr[0], btlbr[0])
    inter_ymin = max(atlbr[1], btlbr[1])
    inter_xmax = min(atlbr[2], btlbr[2])
    inter_ymax = min(atlbr[3], btlbr[3])
    # If there is no overlap
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    # Calculate area of overlap and area of each bounding box
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (atlbr[2] - atlbr[0]) * (atlbr[3] - atlbr[1])
    area2 = (btlbr[2] - btlbr[0]) * (btlbr[3] - btlbr[1])
    # Calculate IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def bbox_iou_by_box(base_obj: Box, target_obj: Box) -> float:
    inter_xmin = max(base_obj.x1, target_obj.x1)
    inter_ymin = max(base_obj.y1, target_obj.y1)
    inter_xmax = min(base_obj.x2, target_obj.x2)
    inter_ymax = min(base_obj.y2, target_obj.y2)
    # If there is no overlap
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    # Calculate area of overlap and area of each bounding box
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
    area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
    # Calculate IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def bbox_ious(atlbrs: List[np.ndarray], btlbrs: List[np.ndarray]) -> np.ndarray:
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    ious = np.array([[bbox_iou(atlbr=atlbr, btlbr=btlbr) for btlbr in btlbrs] for atlbr in atlbrs])
    return ious

def iou_distance(atracks: List[STrack], btracks: List[STrack]):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = bbox_ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def find_most_relevant_object(
    base_obj: Box,
    target_objs: List[Box],
) -> Box:
    most_relevant_obj: Box = None
    best_iou = 0.0
    best_distance = float('inf')
    for target_obj in target_objs:
        if target_obj is not None and not target_obj.is_used:
            # IoUの計算
            iou: float = \
                bbox_iou_by_box(
                    base_obj=base_obj,
                    target_obj=target_obj,
                )
            if iou > best_iou:
                most_relevant_obj = target_obj
                best_iou = iou
                # baseの中心座標とtargetの中心座標のユークリッド距離を計算
                best_distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
            elif  iou > 0.0 and iou == best_iou:
                # baseの中心座標とtargetの中心座標のユークリッド距離を計算
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                if distance < best_distance:
                    most_relevant_obj = target_obj
                    best_distance = distance
    if most_relevant_obj:
        most_relevant_obj.is_used = True
    return most_relevant_obj

def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted([str(file) for file in image_files])

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.
    If the folder does not exist, it is created.

    :param url: URL of the file to download.
    :param folder: Folder where the file will be saved.
    :param filename: Filename to save the file.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Full path for the file
    file_path = os.path.join(folder, filename)
    # Download the file
    print(f"{Color.GREEN('Downloading...')} {url} to {file_path}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{Color.GREEN('Download completed:')} {file_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def get_nvidia_gpu_model() -> List[str]:
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)

        # Extract GPU model numbers using regular expressions
        models = re.findall(r'GPU \d+: (.*?)(?= \(UUID)', output)
        return models
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_cv_color(classid: int) -> Tuple[int, int, int]:
    if classid == 0:
        return (255, 0, 0) # Blue
    elif classid == 1:
        return (0, 255, 0) # Green
    elif classid == 2:
        return (0, 0, 255) # Red
    elif classid == 3:
        return (0,233,245) # Yellow
    else:
        return (255, 255, 255) # Black

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)


def main():
    parser = ArgumentParser()

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 2:
            raise ArgumentTypeError(f"Invalid Value: {ivalue}. Please specify an integer of 2 or greater.")
        return ivalue

    parser.add_argument(
        '-odm',
        '--object_detection_model',
        type=str,
        default='yolov9_e_wholebody28_refine_post_0100_1x3x480x640.onnx',
        help='ONNX/TFLite file path for YOLOv9.',
    )
    parser.add_argument(
        '-gm',
        '--gazelle_model',
        type=str,
        default='gazelle_dinov2_vitb14_inout_1x3x448x448_1xNx4.onnx',
        help='ONNX/TFLite file path for Gaze-LLE.',
    )
    parser.add_argument(
        '-bfem',
        '--body_feature_extractor_model',
        type=str,
        default='mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx',
        choices=[
            'mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x288x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x320x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x352x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x384x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x256x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x288x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x320x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x352x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x384x128_post_feature_only.onnx',
        ],
        help='ONNX/TFLite file path for FastReID.',
    )
    parser.add_argument(
        '-ffem',
        '--face_feature_extractor_model',
        type=str,
        default='face-reidentification-retail-0095_NMx3x128x128_post_feature_only.onnx',
        choices=[
            'face-reidentification-retail-0095_NMx3x128x128_post_feature_only.onnx',
        ],
        help='ONNX/TFLite file path for FaceReID.',
    )
    group_v_or_i = parser.add_mutually_exclusive_group(required=True)
    group_v_or_i.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
        help='Video file path or camera index.',
    )
    group_v_or_i.add_argument(
        '-i',
        '--images_dir',
        type=str,
        help='jpg, png images folder path.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cuda',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-it',
        '--inference_type',
        type=str,
        choices=['fp16', 'int8'],
        default='fp16',
        help='Inference type. Default: fp16',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    parser.add_argument(
        '-dwk',
        '--disable_waitKey',
        action='store_true',
        help=\
            'Disable cv2.waitKey(). '+
            'When you want to process a batch of still images, '+
            ' disable key-input wait and process them continuously.',
    )
    parser.add_argument(
        '-ost',
        '--object_socre_threshold',
        type=float,
        default=0.35,
        help=\
            'The detection score threshold for object detection. Default: 0.35',
    )
    parser.add_argument(
        '-ast',
        '--attribute_socre_threshold',
        type=float,
        default=0.75,
        help=\
            'The attribute score threshold for object detection. Default: 0.70',
    )
    parser.add_argument(
        '-kst',
        '--keypoint_threshold',
        type=float,
        default=0.25,
        help=\
            'The keypoint score threshold for object detection. Default: 0.25',
    )
    parser.add_argument(
        '-kdm',
        '--keypoint_drawing_mode',
        type=str,
        choices=['dot', 'box', 'both'],
        default='dot',
        help='Key Point Drawing Mode. Default: dot',
    )
    parser.add_argument(
        '-cst',
        '--centroid_socre_threshold',
        type=float,
        default=0.30,
        help=\
            'The heatmap centroid score threshold. Default: 0.30',
    )
    parser.add_argument(
        '-dnm',
        '--disable_generation_identification_mode',
        action='store_true',
        help=\
            'Disable generation identification mode. (Press N on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dgm',
        '--disable_gender_identification_mode',
        action='store_true',
        help=\
            'Disable gender identification mode. (Press G on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dlr',
        '--disable_left_and_right_hand_identification_mode',
        action='store_true',
        help=\
            'Disable left and right hand identification mode. (Press H on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhm',
        '--disable_headpose_identification_mode',
        action='store_true',
        help=\
            'Disable HeadPose identification mode. (Press P on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dah',
        '--disable_attention_heatmap_mode',
        action='store_true',
        help=\
            'Disable Attention Heatmap mode. (Press A on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-drc',
        '--disable_render_classids',
        type=int,
        nargs="*",
        default=[],
        help=\
            'Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19',
    )
    parser.add_argument(
        '-efm',
        '--enable_face_mosaic',
        action='store_true',
        help=\
            'Enable face mosaic.',
    )
    parser.add_argument(
        '-oyt',
        '--output_yolo_format_text',
        action='store_true',
        help=\
            'Output YOLO format texts and images.',
    )
    parser.add_argument(
        '-bblw',
        '--bounding_box_line_width',
        type=check_positive,
        default=2,
        help=\
            'Bounding box line width. Default: 2',
    )
    parser.add_argument(
        '-fm',
        '--face_mosaic',
        action='store_true',
        help='Face mosaic.',
    )
    args = parser.parse_args()

    # runtime check
    object_detection_model_file: str = args.object_detection_model
    gazelle_model_file:  str = args.gazelle_model
    body_feature_extractor_model_file: str = args.body_feature_extractor_model
    face_feature_extractor_model_file: str = args.face_feature_extractor_model
    object_detection_model_dir_path = os.path.dirname(os.path.abspath(object_detection_model_file))
    object_detection_model_ext: str = os.path.splitext(object_detection_model_file)[1][1:].lower()
    body_feature_extractor_model_ext: str = os.path.splitext(body_feature_extractor_model_file)[1][1:].lower()
    face_feature_extractor_model_ext: str = os.path.splitext(face_feature_extractor_model_file)[1][1:].lower()
    runtime: str = None
    execution_provider: str = args.execution_provider
    if object_detection_model_ext != body_feature_extractor_model_ext \
        or object_detection_model_ext != face_feature_extractor_model_ext:
        print(Color.RED('ERROR: object_detection_model and face_detection_model and feature_extractor_model must be files with the same extension.'))
        sys.exit(0)
    if object_detection_model_ext == 'onnx':
        err_msg = ''
        if not is_package_installed('onnx') or \
            not is_package_installed('onnxruntime'):
            err_msg = f'onnx onnxruntime'
        if execution_provider == 'tensorrt' and \
            (
                not is_package_installed('sne4onnx') or \
                not is_package_installed('sor4onnx') or \
                not is_package_installed('onnx_graphsurgeon')
            ):
            err_msg = f'{err_msg} sne4onnx sor4onnx onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com'.lstrip()
        if err_msg:
            print(Color.RED(f'ERROR: {err_msg} is not installed. pip install {err_msg}'))
            sys.exit(0)
        runtime = 'onnx'
    elif object_detection_model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)

    WEIGHT_FOLDER_PATH = '.'
    gpu_models = get_nvidia_gpu_model()
    default_supported_gpu_model = False
    if len(gpu_models) == 1:
        gpu_model = gpu_models[0]
        for target_gpu_model in NVIDIA_GPU_MODELS_CC:
            if target_gpu_model in gpu_model:
                default_supported_gpu_model = True
                break

    # Download object detection onnx
    weight_file = os.path.basename(object_detection_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download object detection tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    # Download BodyReID onnx
    weight_file = os.path.basename(body_feature_extractor_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download BodyReID tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    # Download FaceReID onnx
    weight_file = os.path.basename(face_feature_extractor_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download FaceReID tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    video: str = args.video
    images_dir: str = args.images_dir
    disable_waitKey: bool = args.disable_waitKey
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    keypoint_threshold: float = args.keypoint_threshold
    keypoint_drawing_mode: str = args.keypoint_drawing_mode
    centroid_socre_threshold: float = args.centroid_socre_threshold
    disable_generation_identification_mode: bool = args.disable_generation_identification_mode
    disable_gender_identification_mode: bool = args.disable_gender_identification_mode
    disable_left_and_right_hand_identification_mode: bool = args.disable_left_and_right_hand_identification_mode
    disable_headpose_identification_mode: bool = args.disable_headpose_identification_mode
    disable_attention_heatmap_mode: bool = args.disable_attention_heatmap_mode
    disable_render_classids: List[int] = args.disable_render_classids
    enable_face_mosaic: bool = args.enable_face_mosaic
    output_yolo_format_text: bool = args.output_yolo_format_text
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()
    bounding_box_line_width: int = args.bounding_box_line_width
    providers: List[Tuple[str, Dict] | str] = None

    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        elif inference_type == 'int8':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                    "trt_int8_enable": True,
                    "trt_int8_calibration_table_name": "calibration.flatbuffers",
                }
        else:
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': f'{object_detection_model_dir_path}',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                } | ep_type_params,
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    # Model initialization
    object_detection_model = \
        YOLOv9(
            runtime=runtime,
            model_path=object_detection_model_file,
            obj_class_score_th=object_socre_threshold,
            attr_class_score_th=attribute_socre_threshold,
            keypoint_th=keypoint_threshold,
            providers=providers,
        )
    gazelle_model = \
        GazeLLE(
            runtime=runtime,
            model_path=gazelle_model_file,
            providers=providers,
        )
    body_feature_extractor_model = \
        FastReID(
            runtime=runtime,
            model_path=body_feature_extractor_model_file,
            providers=providers,
        )
    face_feature_extractor_model = \
        FaceReidentificationRetail0095(
            runtime=runtime,
            model_path=face_feature_extractor_model_file,
            providers=providers,
        )
    botsort = \
        BoTSORT(
            object_detection_model=object_detection_model,
            body_feature_extractor_model=body_feature_extractor_model,
            face_feature_extractor_model=face_feature_extractor_model,
            frame_rate=30,
        )

    file_paths: List[str] = None
    cap = None
    video_writer = None
    if images_dir is not None:
        file_paths = list_image_files(dir_path=images_dir)
    else:
        cap = cv2.VideoCapture(
            int(video) if is_parsable_to_int(video) else video
        )
        disable_video_writer: bool = args.disable_video_writer
        if not disable_video_writer:
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                filename='output.mp4',
                fourcc=fourcc,
                fps=cap_fps,
                frameSize=(w, h),
            )

    os.remove("output.json") if os.path.exists("output.json") else None
    face_mosaic: bool = args.face_mosaic
    file_paths_count = -1
    movie_frame_count = 0
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1
    while True:
        image: np.ndarray = None
        if file_paths is not None:
            file_paths_count += 1
            if file_paths_count <= len(file_paths) - 1:
                image = cv2.imread(file_paths[file_paths_count])
            else:
                break
        else:
            res, image = cap.read()
            if not res:
                break
            movie_frame_count += 1

        debug_image = copy.deepcopy(image)
        debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        # YOLOv9
        boxes = \
            object_detection_model(
                image=debug_image,
                disable_generation_identification_mode=False,
                disable_gender_identification_mode=False,
                disable_left_and_right_hand_identification_mode=False,
                disable_headpose_identification_mode=False,
            )
        # Gaze-LLE
        head_boxes = [box for box in boxes if box.classid == 7]
        heatmaps = []
        if len(head_boxes) > 0:
            debug_image, heatmaps = \
                gazelle_model(
                    image=debug_image,
                    head_boxes=head_boxes,
                    disable_attention_heatmap_mode=disable_attention_heatmap_mode,
                )

        # 重心 (centroid) を計算する関数
        def calculate_centroid(heatmap: np.ndarray) -> Tuple[int, int, float]:
            """calculate_centroid

            Parameters
            ----------
            heatmap: np.ndarray
                One-channel entire image. [H, W]

            Returns
            -------
            x: int
                Peak X coordinate of the heat map score
            y: int
                Peak Y coordinate of the heat map score
            score: float
                Peak score for heat map score
            """
            # 1. ピーク値を求める
            max_index = np.argmax(heatmap)
            # 2. 1Dインデックスを2Dインデックス (y, x) に変換
            y, x = np.unravel_index(max_index, heatmap.shape)
            return int(x), int(y), heatmap[y, x]

        gazes = []
        for head_box, heatmap in zip(head_boxes, heatmaps):
            cx, cy, score = calculate_centroid(heatmap)
            if score >= centroid_socre_threshold:
                gazes.append(Gaze(head_box.trackid, head_box.cx, head_box.cy, cx, cy))
        # Bot-SORT
        stracks = botsort.update(image=debug_image, detected_boxes=boxes)
        elapsed_time = time.perf_counter() - start_time

        # Writing the dictionary to a JSON file
        def func(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, deque):
                return list(obj)
            return obj.__dict__

        dump = {
            'frame_no': movie_frame_count,
            'elapsed_time': elapsed_time,
            'boxes': boxes,
            'gazes': gazes,
            #'stracks': stracks
        }
        try:
            with open(f"output.json", "a", encoding="utf-8") as json_file:
                # Use json.dump() to write the boxes to the file
                json.dump(dump, json_file, default=func, separators=(',', ':'), ensure_ascii=False)
                json_file.write("\n")
            #print(f"Data successfully written to output.json")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Draw FPS
        if file_paths is None:
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw bounding boxes
        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)

            if classid in disable_render_classids:
                continue

            if classid == 0:
                # Body
                if not disable_gender_identification_mode:
                    # Body
                    if box.gender == 0:
                        # Male
                        color = (255,0,0)
                    elif box.gender == 1:
                        # Female
                        color = (139,116,225)
                    else:
                        # Unknown
                        color = (0,200,255)
                else:
                    # Body
                    color = (0,200,255)
            elif classid == 5:
                # Body-With-Wheelchair
                color = (0,200,255)
            elif classid == 6:
                # Body-With-Crutches
                color = (83,36,179)
            elif classid == 7:
                # Head
                if not disable_headpose_identification_mode:
                    color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216,67,21)
                else:
                    color = (0,0,255)
            elif classid == 16:
                # Face
                color = (0,200,255)
            elif classid == 17:
                # Eye
                color = (255,0,0)
            elif classid == 18:
                # Nose
                color = (0,255,0)
            elif classid == 19:
                # Mouth
                color = (0,0,255)
            elif classid == 20:
                # Ear
                color = (203,192,255)
            elif classid == 21:
                # Shoulder
                color = (255,0,0)
            elif classid == 22:
                # Elbow
                color = (0,255,0)
            elif classid == 23:
                if not disable_left_and_right_hand_identification_mode:
                    # Hands
                    if box.handedness == 0:
                        # Left-Hand
                        color = (0,128,0)
                    elif box.handedness == 1:
                        # Right-Hand
                        color = (255,0,255)
                    else:
                        # Unknown
                        color = (0,255,0)
                else:
                    # Hands
                    color = (0,255,0)
            elif classid == 26:
                # Knee
                color = (0,0,255)
            elif classid == 27:
                # Foot
                color = (250,0,136)

            if (classid == 0 and not disable_gender_identification_mode) \
                or (classid == 7 and not disable_headpose_identification_mode) \
                or (classid == 23 and not disable_left_and_right_hand_identification_mode) \
                or classid == 16 \
                or classid in [21, 22, 26]:

                # Body
                if classid == 0:
                    if box.gender == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Head
                elif classid == 7:
                    if box.head_pose == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Face
                elif classid == 16:
                    if enable_face_mosaic:
                        w = int(abs(box.x2 - box.x1))
                        h = int(abs(box.y2 - box.y1))
                        small_box = cv2.resize(debug_image[box.y1:box.y2, box.x1:box.x2, :], (3,3))
                        normal_box = cv2.resize(small_box, (w,h))
                        if normal_box.shape[0] != abs(box.y2 - box.y1) \
                            or normal_box.shape[1] != abs(box.x2 - box.x1):
                                normal_box = cv2.resize(small_box, (abs(box.x2 - box.x1), abs(box.y2 - box.y1)))
                        debug_image[box.y1:box.y2, box.x1:box.x2, :] = normal_box
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Hands
                elif classid == 23:
                    if box.handedness == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Shoulder, Elbow, Knee
                elif classid in [21, 22, 26]:
                    if keypoint_drawing_mode in ['dot', 'both']:
                        cv2.circle(debug_image, (box.cx, box.cy), 5, (255,255,255), -1)
                        cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
                    if keypoint_drawing_mode in ['box', 'both']:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)

            else:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

            # Attributes text
            trackid_text = ''
            if box.trackid != 0:
                trackid_text = f'{box.trackid}:'

            generation_txt = ''
            if box.generation == -1:
                generation_txt = ''
            elif box.generation == 0:
                generation_txt = 'Adult'
            elif box.generation == 1:
                generation_txt = 'Child'

            gender_txt = ''
            if box.gender == -1:
                gender_txt = ''
            elif box.gender == 0:
                gender_txt = 'M'
            elif box.gender == 1:
                gender_txt = 'F'

            attr_txt = f'{trackid_text}{generation_txt}({gender_txt})' if gender_txt != '' else f'{trackid_text}{generation_txt}'

            headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
            attr_txt = f'{attr_txt} {headpose_txt}' if headpose_txt != '' else f'{attr_txt}'

            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            handedness_txt = ''
            if box.handedness == -1:
                handedness_txt = ''
            elif box.handedness == 0:
                handedness_txt = 'L'
            elif box.handedness == 1:
                handedness_txt = 'R'
            cv2.putText(
                debug_image,
                f'{trackid_text}{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{trackid_text}{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )

        # for strack in stracks:
        #     color = (255,0,0)
        #     #cv2.rectangle(debug_image, (int(strack.tlbr[0]), int(strack.tlbr[1])), (int(strack.tlbr[2]), int(strack.tlbr[3])), (255,255,255), 2)
        #     #cv2.rectangle(debug_image, (int(strack.tlbr[0]), int(strack.tlbr[1])), (int(strack.tlbr[2]), int(strack.tlbr[3])), color, 1)
        #     ptx = int(strack.tlbr[0]) if int(strack.tlbr[0])+50 < debug_image_w else debug_image_w-50
        #     pty = int(strack.tlbr[1])-10 if int(strack.tlbr[1])-25 > 0 else 20
        #     cv2.putText(debug_image, f'{strack.track_id}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #     cv2.putText(debug_image, f'{strack.track_id}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        #     if strack.body is not None:
        #         body = strack.body
        #         if body.head is not None:
        #             color = get_cv_color(body.head.classid)
        #             #cv2.rectangle(debug_image, (body.head.x1, body.head.y1), (body.head.x2, body.head.y2), (255,255,255), 2)
        #             #cv2.rectangle(debug_image, (body.head.x1, body.head.y1), (body.head.x2, body.head.y2), color, 1)
        #             ptx = body.head.x1 if body.head.x1+50 < debug_image_w else debug_image_w-50
        #             pty = body.head.y1-10 if body.head.y1-25 > 0 else 20
        #             cv2.putText(debug_image, f'{body.head.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #             cv2.putText(debug_image, f'{body.head.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        #             if body.head.face is not None:
        #                 color = get_cv_color(body.head.face.classid)
        #                 if face_mosaic:
        #                     w = abs(body.head.face.x2 - body.head.face.x1)
        #                     h = abs(body.head.face.y2 - body.head.face.y1)
        #                     debug_image[body.head.face.y1:body.head.face.y2, body.head.face.x1:body.head.face.x2, :] = \
        #                         cv2.resize(cv2.resize(debug_image[body.head.face.y1:body.head.face.y2, body.head.face.x1:body.head.face.x2, :], (2, 2)), (w, h))
        #                 draw_dashed_rectangle(debug_image, (body.head.face.x1, body.head.face.y1), (body.head.face.x2, body.head.face.y2), (255,255,255), 2, 5)
        #                 draw_dashed_rectangle(debug_image, (body.head.face.x1, body.head.face.y1), (body.head.face.x2, body.head.face.y2), color, 1, 5)
        #                 ptx = body.head.face.x1 if body.head.face.x1+50 < debug_image_w else debug_image_w-50
        #                 pty = body.head.face.y1-10 if body.head.face.y1-25 > 0 else 20
        #                 cv2.putText(debug_image, f'{body.head.face.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #                 cv2.putText(debug_image, f'{body.head.face.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        #         if body.hand1 is not None:
        #             color = get_cv_color(body.hand1.classid)
        #             #cv2.rectangle(debug_image, (body.hand1.x1, body.hand1.y1), (body.hand1.x2, body.hand1.y2), (255,255,255), 2)
        #             #cv2.rectangle(debug_image, (body.hand1.x1, body.hand1.y1), (body.hand1.x2, body.hand1.y2), color, 1)
        #             ptx = body.hand1.x1 if body.hand1.x1+50 < debug_image_w else debug_image_w-50
        #             pty = body.hand1.y1-10 if body.hand1.y1-25 > 0 else 20
        #             cv2.putText(debug_image, f'{body.hand1.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #             cv2.putText(debug_image, f'{body.hand1.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        #         if body.hand2 is not None:
        #             color = get_cv_color(body.hand2.classid)
        #             #cv2.rectangle(debug_image, (body.hand2.x1, body.hand2.y1), (body.hand2.x2, body.hand2.y2), (255,255,255), 2)
        #             #cv2.rectangle(debug_image, (body.hand2.x1, body.hand2.y1), (body.hand2.x2, body.hand2.y2), color, 1)
        #             ptx = body.hand2.x1 if body.hand2.x1+50 < debug_image_w else debug_image_w-50
        #             pty = body.hand2.y1-10 if body.hand2.y1-25 > 0 else 20
        #             cv2.putText(debug_image, f'{body.hand2.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #             cv2.putText(debug_image, f'{body.hand2.trackid}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        # Drawing of the point of gaze
        for gaze in gazes:
            cv2.line(debug_image, (gaze.head_x, gaze.head_y), (gaze.target_x, gaze.target_y), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.line(debug_image, (gaze.head_x, gaze.head_y), (gaze.target_x, gaze.target_y), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(debug_image, (gaze.target_x, gaze.target_y), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(debug_image, (gaze.target_x, gaze.target_y), 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            if gaze.trackid != 0:
                cv2.putText(debug_image, f'{gaze.trackid}', (gaze.target_x, gaze.target_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(debug_image, f'{gaze.trackid}', (gaze.target_x, gaze.target_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        if file_paths is not None:
            basename = os.path.basename(file_paths[file_paths_count])
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{basename}', debug_image)

        if file_paths is not None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_i.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_o.png', debug_image)
            with open(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        elif file_paths is None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{movie_frame_count:08d}.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_i.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_o.png', debug_image)
            with open(f'output/{movie_frame_count:08d}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        if video_writer is not None:
            video_writer.write(debug_image)
            # video_writer.write(image)

        cv2.imshow("test", debug_image)

        key = cv2.waitKey(1) if file_paths is None or disable_waitKey else cv2.waitKey(0)
        if key == 27: # ESC
            break
        elif key == 110: # N, Generation mode switch
            disable_generation_identification_mode = not disable_generation_identification_mode
        elif key == 103: # G, Gender mode switch
            disable_gender_identification_mode = not disable_gender_identification_mode
        elif key == 112: # P, HeadPose mode switch
            disable_headpose_identification_mode = not disable_headpose_identification_mode
        elif key == 104: # H, HandsLR mode switch
            disable_left_and_right_hand_identification_mode = not disable_left_and_right_hand_identification_mode
        elif key == 107: # K, Keypoints mode switch
            if keypoint_drawing_mode == 'dot':
                keypoint_drawing_mode = 'box'
            elif keypoint_drawing_mode == 'box':
                keypoint_drawing_mode = 'both'
            elif keypoint_drawing_mode == 'both':
                keypoint_drawing_mode = 'dot'
        elif key == 97: # A, mode switch
            disable_attention_heatmap_mode = not disable_attention_heatmap_mode

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
