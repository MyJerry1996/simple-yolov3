from __future__ import division
import time

import torch

from models.darknet import create_model
from models.detector import get_detector
from util import *
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import random
from loguru import logger
import yaml
from torch.utils.data.dataloader import DataLoader
from dataset import COCODataset, COCO128Dataset
from loss import ComputeLoss
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter

RANK = int(os.getenv('RANK', -1))


# 保存模型
def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def arg_parse():
    """
    Parse arguments to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="D:\\allMyCode\\pycharmProj\\datasets\\coco128", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to", default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=16, type=int)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5,
                        type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.1, type=float)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default=None, type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416,
                        type=int)
    parser.add_argument("--lr", dest='lr', help="learning rate", default=0.000125, type=float)
    parser.add_argument("--num_workers", dest='num_workers', help="num of workers", default=4, type=int)
    parser.add_argument("--num_epochs", dest='num_epochs', help="num of epochs", default=150, type=int)
    parser.add_argument("--lr_step", dest='lr_step', help="decrease lr when epoch equals to step", default=100,
                        type=int)
    parser.add_argument("--momentum", dest='momentum', help="momentum", default=0.937, type=int)
    parser.add_argument("--gpus", dest='gpus', default='0', help="-1 for CPU, use comma for multiple gpus")
    parser.add_argument("--gamma", dest='gamma', default=0.9, help="the hyperparams in focal loss")
    parser.add_argument("--hyp", type=str, default=ROOT / 'hyp.yaml', help='hyperparameters path')
    parser.add_argument("--save_interval", type=int, default=5, help='save the model in defined epoch interval')
    parser.add_argument("--num_classes", type=int, default=80, help='num of classes')

    return parser.parse_args()


@logger.catch
def train(args, dataloader, model, optimizer, device, anchors, hyp, start_epoch=1):
    # start training
    # set_device(gpus, chunk_sizes, device)
    logger.info("Let's start training!")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        if args.weightsfile:
            model.load_state_dict(torch.load(args.weightsfile))

        writer = SummaryWriter()

        num_iters = len(dataloader)
        it = enumerate(dataloader)
        bar = tqdm(it, total=num_iters, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        compute_loss = ComputeLoss(model, hyp, anchors)

        for i, (inputs, labels) in bar:
            inputs = torch.Tensor([t.numpy() for t in inputs])
            if i >= num_iters:
                break
            # inputs: tensor[B, C, H, W]   labels: list[[x1, y1, w1, h1, cls1], [x2, y2, w2, h2, cls2], ...]
            # inputs, labels = data
            # pred: tensor[B, (13*13+26*26+52*52)*3, (xywh+confidence+cls)] [1, 10647, 85]

            # output = nms(pred, args.nms_thresh)
            # for i in range(pred.shape[0]):  # NMS module pre feats in batches
            #     bboxes = pred[i, :, :5].squeeze(0)
            #     nms_output.append(nms(bboxes, args.nms_thresh))
            # loss_sum = 0.0  # init loss class
            # loss_items_sum = torch.zeros(3)

            # pred = nms(pred, args.nms_thresh)
            # for batch in range(args.bs):


            pred = model(inputs / 255)
            # loss, loss_items = compute_loss(pred, labels[batch].to(device))
            loss, loss_items = compute_loss(pred, labels)
            lbox, lobj, lcls = loss_items[0], loss_items[1], loss_items[2]

            # data into logger
            writer.add_scalar('loss/train', loss, epoch)
            writer.add_scalar('lbox/train', lbox, epoch)
            writer.add_scalar('lobj/train', lobj, epoch)
            writer.add_scalar('lcls/train', lcls, epoch)
            logger.info(
                'loss: {:2.3f} | lbox: {:2.3f} | lobj: {:2.3f} | lcls: {:2.3f}'.format(loss[0], lbox, lobj, lcls))

            # update optimizer and backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save model
        try:
            if epoch % args.save_interval == 0:
                save_model(args.model_path, epoch, model, optimizer=optimizer)
        except:
            print("ERROR occurred when saving model")

        torch.cuda.empty_cache()


def main(args):
    # load args
    images = args.images
    batch_size = args.bs
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    num_classes = args.num_classes
    device = torch.device('cuda' if int(args.gpus[0]) >= 0 else 'cpu')
    anchors = torch.tensor([116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119, 10, 13, 16, 30, 33, 23],
                           device=device)
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    # setting up dataset, model and optimizer
    backbone = create_model(num_classes)
    detector = get_detector(416, num_classes, anchors=anchors)
    model = nn.Sequential(backbone, detector)
    model = nn.DataParallel(model, device_ids=[0])
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # dataset = COCODataset(images)
    dataset = COCO128Dataset(images)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              sampler=None,
                              collate_fn=COCO128Dataset.collate_fn,
                              pin_memory=True)

    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            sampler=None,
                            pin_memory=True)
    logger.info('dataset is ready!')

    # train
    start_epoch = 1
    train(args, train_loader, model, optimizer, device, anchors, hyp, start_epoch)


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    args = arg_parse()
    main(args)
