from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
import torch.nn as nn
import os


# 实现的功能：train模型，log记录，loss输出搞清楚，

# class ModelWithLoss(torch.nn.Module):
#     def __init__(self, model, loss):
#         super(ModelWithLoss, self).__init__()
#         self.model = model
#         self.loss = loss
#
#     def forward(self, batch):
#         outputs = self.model(batch['input'])
#         loss, loss_stats = self.loss(outputs, batch)
#         return outputs[-1], loss, loss_stats
#
#
# class Trainer(object):
#     def __init__(
#             self, args, optimizer=None):
#         self.opt = args
#         self.optimizer = optimizer
#         self.loss = torch.nn.CrossEntropyLoss()
#
#     def run_epoch(self, phase, epoch, data_loader):
#         model_with_loss = self.model_with_loss
#         if phase == 'train':
#             model_with_loss.train()
#         else:
#             if len(self.opt.gpus) > 1:
#                 model_with_loss = self.model_with_loss.module
#             model_with_loss.eval()
#             torch.cuda.empty_cache()
#
#         args = self.args
#         results = {}
#
#     def _get_loss(self, args):
#         loss_states = ['loss']
#
#     def train(self, epoch, data_loader):
#         return self.run_epoch('train', epoch, data_loader)
#         pass
#
#     def val(self):
#         pass


class Trainer(object):
    def __init__(self, cfg, model, accumulate=1, multi_scale=False, freeze_backbone=False, transfer=False):
        self.cfg = cfg
        self.model = model
        self.opti = nn.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'],
                                 weight_decay=hyp['weight_decay'])

    def train(self):
        # device = torch_utils.select_device()
        weights = 'weights' + os.sep   #python是跨平台的。在Windows上，文件的路径分隔符是'\'，在Linux上是'/'。
        latest = weights + 'latest.pt'
        model = self.model.to(device)
        opti = self.opti.to(device)
        cutoff = -1  # backbone reaches to cutoff layer
        start_epoch = 0
        best_loss = float('inf')
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
        if self.cfg.resume:
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

            start_epoch = chkpt['epoch'] + 1

            start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                opti.load_state_dict(chkpt['optimizer'])
                best_loss = chkpt['best_loss']
            del chkpt

        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

            lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inv exp ramp to lr0 * 1e-2
            scheduler = nn.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)



    def compute_loss(p, targets, model):  # predictions, targets, model
        ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lxy, lwh, lcls, lconf = ft([0]), ft([0]), ft([0]), ft([0])
    txy, twh, tcls, indices = build_targets(model, targets)#在13 26 52维度中找到大于iou阈值最适合的anchor box 作为targets
    #txy[维度(0:2),(x,y)] twh[维度(0:2),(w,h)] indices=[0,anchor索引，gi，gj]

    # Define criteria
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()

    # Compute losses
    h = model.hyp  # hyperparameters
    bs = p[0].shape[0]  # batch size
    k = h['k'] * bs  # loss gain
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
        tconf = torch.zeros_like(pi0[..., 0])  # conf


        # Compute losses
        if len(b):  # number of targets
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors 找到p中与targets对应的数据lxy
            tconf[b, a, gj, gi] = 1  # conf
            # pi[..., 2:4] = torch.sigmoid(pi[..., 2:4])  # wh power loss (uncomment)

            lxy += (k * h['xy']) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
            lwh += (k * h['wh']) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss
            lcls += (k * h['cls']) * CE(pi[..., 5:], tcls[i])  # class_conf loss

        # pos_weight = ft([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * h['conf']) * BCE(pi0[..., 4], tconf)  # obj_conf loss
    loss = lxy + lwh + lconf + lcls

    return loss, torch.cat((lxy, lwh, lconf, lcls, loss)).detach()