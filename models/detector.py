from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from util import *


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def get_output_and_grid(output, inp_size, num_classes, anchors, CUDA=True):
    """

    :param output: output from create_model(input_size)
    :param inp_size: size of input
    :param num_classes: ...
    :param anchors: list of anchors size
    :param CUDA: Enable Flag of GPUs
    :return:
    """
    batch_size = output.shape[0]
    stride = inp_size // output.size(2)
    grid_size = inp_size // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    output = output.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    output[:, :, 0] = torch.sigmoid(output[:, :, 0])
    output[:, :, 1] = torch.sigmoid(output[:, :, 1])
    output[:, :, 4] = torch.sigmoid(output[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    output[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    output[:, :, 2:4] = torch.exp(output[:, :, 2:4]) * anchors

    output[:, :, 5: 5 + num_classes] = torch.sigmoid((output[:, :, 5: 5 + num_classes]))

    output[:, :, :4] *= stride

    return output


class YOYOv3DetHead(nn.Module):  # inputs: list of three outputs from darknet
    def __init__(self, inp_dim, num_classes, anchors, istrain):
        super(YOYOv3DetHead, self).__init__()
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.anchors = [anchors[:3], anchors[3:6], anchors[6:]]

        self.istrain = istrain

        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        self.anchors = [anchors[:3], anchors[3:6], anchors[6:]]

    def forward(self, inputs):
        prediction = []
        for i in range(len(inputs)):
            # structure of inputs:
            # yolo layer1，x:13*13*255，
            # mask= [6, 7, 8] anchors= [(116, 90), (156, 198), (373, 326)]
            # yolo layer2，x:26*26*255，
            #  mask= [3, 4, 5] anchors= [(30, 61), (62, 45), (59, 119)]
            # yolo layer3，x:52*52*255，
            #  mask= [0, 1, 2] anchors= [(10, 13), (16, 30), (33, 23)]
            # x = get_output_and_grid(inputs[i], self.inp_dim, self.num_classes, self.anchors)
            anchors = self.anchors[i]
            anchors = torch.FloatTensor(anchors)

            anchors = anchors.cuda()
            # print(inputs[i].shape)

            x = predict_transform(inputs[i], self.inp_dim, anchors, self.num_classes, CUDA=True)
            x = x.cpu()

            prediction.append(x)

        return prediction  # output -> list:(B, 3, 13, 13, 85),(B, 3, 26, 26, 85),(B, 3, 52, 52, 85)


def get_detector(inp_dim, num_classes, anchors, istrain=True):
    return YOYOv3DetHead(inp_dim, num_classes, anchors, istrain)
