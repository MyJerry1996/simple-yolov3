from __future__ import division

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import os


######################################################################
# nms part
######################################################################


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def unique(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


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


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # 将confidence过低的prediction用mask过滤掉
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False
    # 首先将网络输出方框属性(x,y,w,h)转换为在网络输入图片(416x416)坐标系中,方框左上角与右下角坐标(x1,y1,x2,y2)，以方便NMS操作
    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        # confidence threshholding
        # NMS

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
        #

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)  # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0
    # 最终返回的output一行为一个bbox：(x1,y1,x2,y2,s,s_cls,index_cls)，size=7，前四个是bbox左上角和右下角的坐标，
    # s是这个方框含有目标的得分，s_cls是这个方框中所含目标最有可能的类别的概率得分


######################################################################
# data prep part
######################################################################


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    if CUDA:
        prediction = prediction.cuda()
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, num_anchors, grid_size, grid_size, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the centre_X, centre_Y. and object confidence
    prediction[..., 0] = torch.sigmoid(prediction[..., 0])
    prediction[..., 1] = torch.sigmoid(prediction[..., 1])
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(prediction.shape[0], prediction.shape[1], 1)
    x_y_offset = x_y_offset.view(batch_size,
                                 num_anchors,
                                 grid_size,
                                 grid_size,
                                 -1)

    # coordinate of grid point + xyoffset = true centre bbox_xy
    prediction[..., :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors) / stride

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(batch_size * grid_size * grid_size, 1).reshape(batch_size,
                                                                            num_anchors,
                                                                            grid_size,
                                                                            grid_size,
                                                                            -1)
    # decode bbox_wh
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * anchors
    # class flag
    prediction[..., 5: 5 + num_classes] = torch.sigmoid((prediction[..., 5: 5 + num_classes]))
    # bbox_wh and bbox_xy mul the stride resize to the scale in img 416*416
    prediction[..., :4] *= stride

    return prediction


# def nms(bounding_boxes, threshold):
#     if len(bounding_boxes) == 0:
#         return [], []
#     # bboxes = np.array(bounding_boxes)
#     # score = np.array(confidence_score)
#
#     bboxes = xywh2xyxy(bounding_boxes[:, :4]).detach().numpy()
#     score = bounding_boxes[:, 4].detach().numpy()
#
#     # 计算 n 个候选框的面积大小
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 2]
#     y2 = bboxes[:, 3]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#
#     # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
#     order = np.argsort(score)
#
#     # picked_boxes = []  # 返回值
#     # picked_score = []  # 返回值
#     outputs = []
#     while order.size > 0:
#         # 将当前置信度最大的框加入返回值列表中
#         index = order[-1]
#         # picked_boxes.append(bounding_boxes[index])
#         # picked_score.append(confidence_score[index])
#         outputs.append(bounding_boxes[index])
#
#         # 获取当前置信度最大的候选框与其他任意候选框的相交面积
#         x11 = np.minimum(x1[index], x1[order[:-1]])
#         y11 = np.minimum(y1[index], y1[order[:-1]])
#         x22 = np.maximum(x2[index], x2[order[:-1]])
#         y22 = np.maximum(y2[index], y2[order[:-1]])
#         w = np.maximum(0.0, x22 - x11 + 1)
#         h = np.maximum(0.0, y22 - y11 + 1)
#         intersection = w * h
#         # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
#         ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
#         left = np.where(ratio < threshold)
#         order = order[left]
#
#     return outputs


def nms(predictions, threshold=0.25, conf_thres=0.5):
    outputs = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        xc = prediction[..., 4] > conf_thres  # candidates

        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= threshold <= 1, f'Invalid IoU {threshold}, valid values are between 0.0 and 1.0'

        max_nms = 20000
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]  # filter the low confidence bbox

            # compute class confidence
            x[:, 5:] *= x[:, 4:5]

            box = xywh2xyxy(x[:, :4])

            # 85 -> 6
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # check num of bboxes
            num_bboxes = x.shape[0]
            if not num_bboxes:  # no boxes
                continue
            elif num_bboxes > max_nms:
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            boxes, scores = x[:, :4], x[:, 4]
            i = torchvision.ops.nms(boxes, scores, threshold)

            output[xi] = x[i]

            outputs.append(output)

    return outputs
