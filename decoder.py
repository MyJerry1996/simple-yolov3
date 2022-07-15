from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from util import *


def yolo_correct_bboxes(bbox, scale):
    return [i * scale for i in bbox]


def get_anchors_and_decode(feats, anchors, num_classes, input_shape,
                           calc_loss):  # feat.size: [B, grid_shape[0, 1], 85 * 3]
    num_anchors = len(anchors)
    grid_shape = feats.shape[1:3]

    # generate feature map grid for 3 different anchors (actually the coordinate of each grid point)
    grid_x = torch.arange(0, grid_shape[0]).reshape(1, -1, 1, 1).expand(grid_shape[0], -1, num_anchors, -1)
    grid_y = torch.arange(0, grid_shape[1]).reshape(-1, 1, 1, 1).expand(-1, grid_shape[1], num_anchors, -1)
    grid = torch.cat([grid_x, grid_y], -1)

    # generate anchor bboxes, shape:[13, 13, num_anchors, 2]
    anchors_tensor = anchors.reshape(1, 1, num_anchors, 2).expand(grid_shape[0], grid_shape[1], -1, -1)

    # transform feats into [B, 13, 13, 3, 85]
    feats = feats.reshape(-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5)

    # xyoffset must be sigmoided so that the values are in the interval [0, 1]
    bbox_xy = (F.sigmoid(feats[..., :2]) + grid) / grid_shape[::-1]
    bbox_wh = torch.exp(feats[..., 2:4]) * anchors_tensor / input_shape[::-1]

    # according to decode algorithm the confidence and class flag must go through the sigmoid func
    bbox_confidence = F.sigmoid(feats[..., 4:5])
    bbox_class_probs = F.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, bbox_xy, bbox_wh
    return bbox_xy, bbox_wh, bbox_confidence, bbox_class_probs


def DecodeBox(outputs,
              anchors,
              num_classes,
              image_shape,
              input_shape,
              # -----------------------------------------------------------#
              #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
              #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
              #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
              # -----------------------------------------------------------#
              anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
              max_boxes=100,
              confidence=0.5,
              nms_iou=0.3,
              letterbox_image=False):
    bboxes_xy = []
    bboxes_wh = []
    bboxes_confidence = []
    bboxes_class_probs = []
    for i in range(len(outputs)):
        bbox_xy, bbox_wh, bbox_confidence, bbox_clas_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
