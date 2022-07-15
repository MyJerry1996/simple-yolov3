from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import cv2
import random
import torch
import torch.utils.data as data


def letterbox(img, new_shape=(416, 416), color=(128, 128, 128), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # add border
    return img, ratio, dw, dh


class COCODataset(data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def _coco_box_to_bbox(self, box):  # xywh -> xyxy
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __init__(self, path, img_size=416, letterbox=True, augment=False, max_obj=32):
        # with open(path, "r") as f:
        #     img_files = f.read().splitlines()
        #     self.img_files = list(filter(lambda x: len(x) > 0, img_files))
        #
        # n = len(self.img_files)
        # assert n > 0, 'No images found in %s' % path
        self.img_dir = path
        self.img_size = img_size
        self.augment = augment
        self.annot_path = os.path.join(path, 'annotations/instances_train2017.json')

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.letterbox = letterbox
        self.max_obj = max_obj
        # self.label_files = [
        #     x.replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
        #     for x in self.img_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_id = self.images[index]  # 导入索引值
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']  #
        img_path = os.path.join(self.img_dir, 'train2017', file_name)  # 导入图片路径
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = len(anns)

        # read imgs
        img = cv2.imread(img_path)

        # transform imgs into inp_size
        inp_size = self.img_size
        height, width = img.shape[0], img.shape[1]
        if self.letterbox:
            inp, ratio, dw, dh = letterbox(img, inp_size)

        else:
            inp = cv2.resize(img, (inp_size, inp_size))
        inp = inp.transpose(2, 0, 1)

        # for k in range(num_objs):
        #     ann = anns[k]
        #     bbox = self._coco_box_to_bbox(ann['bbox'])
        #     cls_id = int(ann['category_id']) - 1
        #
        #     bbox[[1, 3]] = bbox[[1, 3]] / width
        #     bbox[[2, 4]] = bbox[[2, 4]] / height  # regularization
        #
        #     bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, inp_size - 1)
        #     bbox[[2, 4]] = np.clip(bbox[[2, 4]], 0, inp_size - 1)
        #     new_bbox = np.zeros(5, dtype=int)
        #     new_bbox[0] = cls_id
        #     new_bbox[1:5] = bbox
        #     labels.append(new_bbox)
        labels = np.zeros((num_objs, 6))
        if num_objs:
            for k in range(min(num_objs, self.max_obj)):
                ann = anns[k]
                bbox = ann['bbox']
                cls_id = int(ann['category_id']) - 1

                # regularization
                if self.letterbox:
                    r = ratio[0]
                    bbox[0] = (bbox[0] * r + dw) / inp_size
                    bbox[1] = (bbox[1] * r + dh) / inp_size
                    bbox[2] = (bbox[2] * r) / inp_size
                    bbox[3] = (bbox[3] * r) / inp_size
                else:
                    bbox[[0, 2]] = bbox[[0, 2]] / width
                    bbox[[1, 3]] = bbox[[1, 3]] / height

                bbox = np.clip(bbox, 0, 1)

                new_bbox = np.zeros(6, dtype=float)
                new_bbox[0] = 0.
                new_bbox[1] = float(cls_id)
                new_bbox[2:6] = bbox
                labels[k, :] = new_bbox

        return torch.from_numpy(inp), torch.Tensor(labels)

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed

        return img, label

    # data augment

    # """
    #
    # :param index: index of img
    # :return: img and label
    # """
    # img_path = self.img_files[index]
    # label_path = self.label_files[index]
    #
    # img = cv2.imread(img_path)
    # assert img is not None, 'File Not Found ' + img_path
    # h, w, _ = img.shape
    # img, ratio, padw, padh = letterbox(img, new_shape=self.img_size)
    #
    # labels = []
    # if os.path.isfile(label_path):
    #     with open(label_path, 'r') as file:
    #         lines = file.read().splitlines()
    #     x = np.array([x.split() for x in lines], dtype=np.float32)
    #     if x.size > 0:
    #         # Normalized xywh to pixel xyxy format
    #         labels = x.copy()
    #         labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
    #         labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
    #         labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
    #         labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh
    #         print(labels)
    #
    # # if self.augment:
    # #     img, labels = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
    #
    # nL = len(labels)
    # if nL:
    #     # convert xyxy to xywh
    #     labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size
    #
    # if self.augment:
    #     # random left-right flip
    #     lr_flip = True
    #     if lr_flip and random.random() > 0.5:
    #         img = np.fliplr(img)
    #         if nL:
    #             labels[:, 1] = 1 - labels[:, 1]
    #
    #     ud_flip = True
    #     if ud_flip and random.random() > 0.5:
    #         img = np.flipud(img)
    #         if nL:
    #             labels[:, 2] = 1 - labels[:, 2]
    #
    # labels_out = torch.zeros((nL, 6))
    # if nL:
    #     labels_out[:, 1:] = torch.from_numpy(labels)
    #
    # # Normalize
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    # img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #
    # return torch.from_numpy(img), labels_out, img_path, (h, w)


# def xyxy2xywh(bboxes):
#     """
#
#     :param bboxes: np.array, bboxes of images
#     """
#     bboxes[:, 2] = bboxes[:, 0] - bboxes[:, 2]
#     bboxes[:, 3] = bboxes[:, 1] - bboxes[:, 3]
#
#     return bboxes

# def random_affine(img, labels, degrees, translate, scale):


class COCO128Dataset(data.Dataset):

    def __init__(self, path, img_size=416, letterbox=True, augment=False):
        self.img_dir = os.path.join(path, "images\\train2017")
        self.label_dir = os.path.join(path, "labels\\train2017")
        self.imgs = os.listdir(self.img_dir)
        self.labels = os.listdir(self.label_dir)

        self.img_size = img_size
        self.letterbox = letterbox
        self.augment = augment
        for i in self.imgs:
            print(i)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        img = cv2.imread(img_path)

        # letterbox
        height, width = img.shape[0], img.shape[1]
        if self.letterbox:
            inp, ratio, dw, dh = letterbox(img, self.img_size)
        else:
            inp = cv2.resize(img, (self.img_size, self.img_size))
        inp = inp.transpose(2, 0, 1)

        # labels
        with open(label_path, "r") as f:
            labels = np.loadtxt(f).reshape(-1, 5)

        num_objs = len(labels)
        labels_out = np.zeros((num_objs, 6))
        if num_objs:
            for k in range(num_objs):
                bbox = labels[k, 1:5]
                cls_id = labels[k, 0]

                # regularization
                if self.letterbox:
                    r = ratio[0]
                    bbox[0] = (bbox[0] * r + dw)
                    bbox[1] = (bbox[1] * r + dh)
                    bbox[2] = (bbox[2] * r)
                    bbox[3] = (bbox[3] * r)
                else:
                    bbox[[0, 2]] = bbox[[0, 2]] / width
                    bbox[[1, 3]] = bbox[[1, 3]] / height

                bbox = np.clip(bbox, 0, 1)

                labels_out[k, 0] = 0.
                labels_out[k, 1] = float(cls_id)
                labels_out[k, 2:6] = bbox

        return torch.from_numpy(inp), torch.from_numpy(labels_out),

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)
