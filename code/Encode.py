import numpy as np
import os
import paddle
import cv2
import json
import random
import time
import copy


class Encode():
    def __init__(self):
        self.divide = [80, 160, 320, 640]
        self.stride = [8, 16, 32, 64, 128]
        self.size2layer = [[125, 306], [63, 153], [32, 77], [16, 39], [8, 20]]
        self.layerNum = 5
        self.clsNum = 20

    def gt_process(self, gt):
        if gt is None:
            return [np.zeros((0, 5))] * 5
        meanedge = np.mean((gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]), axis=0)
        gt1 = gt[meanedge <= 100]
        gt2 = gt[(meanedge <= 200) * (meanedge >= 80)]
        gt3 = gt[(meanedge <= 400) * (meanedge >= 160)]
        gt4 = gt[(meanedge <= 800) * (meanedge >= 320)]
        gt5 = gt[meanedge >= 640]
        return [gt1, gt2, gt3, gt4, gt5]

    def encode(self, gt):
        # gt为 numpy形式, shape = (N, 5) loc = x1,y1,x2,y2,cls
        # 为gt 生成目标图 centerness, 分类 和 坐标
        gt = self.gt_process(gt)  # gt为list,存储5个layer的gt
        Reg = np.zeros(shape=(0, 4))
        Center = np.zeros(shape=(0,))
        Cls = np.zeros(shape=(0, self.clsNum))
        for i in range(self.layerNum):
            # if(gt[i].shape[0] == 0):continue
            targetReg = np.zeros(shape=self.size2layer[i] + [4], dtype=np.float32)  # 4分别为（l, r, t, b）
            targetCenter = np.zeros(shape=self.size2layer[i], dtype=np.float32)
            targetCls = np.zeros(shape=self.size2layer[i] + [self.clsNum], dtype=np.float32)
            gt_cls = gt[i][:, 4].astype(np.int32)
            # print("layer = %d, gt[i] = %s"%(i+1, gt[i]))
            targetGt = (gt[i][:, :4] / self.stride[i])
            targetGt[:, [2, 3]] = np.ceil(targetGt[:, [2, 3]])
            targetGt[:, [0, 1]] = np.floor(targetGt[:, [0, 1]])
            targetGt = targetGt.astype(np.int32)
            # [w, h] = targetGt[:,2] - targetGt[:,0] + 1, targetGt[:,3] - targetGt[:,1] + 1
            w, h = targetGt[:, 2] - targetGt[:, 0], targetGt[:, 3] - targetGt[:, 1]
            for j in range(targetGt.shape[0]):
                bbox = targetGt[j]
                bboxMap = np.zeros(shape=(h[j], w[j], 4))
                bboxMap[:, :, 0] = np.arange(w[j]).reshape(1, -1) * np.ones(shape=(h[j], 1))
                bboxMap[:, :, 1] = w[j] - 1 - bboxMap[:, :, 0]
                bboxMap[:, :, 2] = np.arange(h[j]).reshape(-1, 1) * np.ones(shape=(1, w[j]))
                bboxMap[:, :, 3] = h[j] - 1 - bboxMap[:, :, 2]
                bboxMap += 1
                # try:
                targetReg[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bboxMap
                # except:
                #    print("bbox ",bbox)
                #    print("targetReg ",targetReg.shape)
                #    print("bboxMap ",bboxMap.shape)
                targetCenter[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.sqrt(
                    (np.minimum(bboxMap[:, :, 0], bboxMap[:, :, 1]) * np.minimum(bboxMap[:, :, 2], bboxMap[:, :, 3])) /
                    (np.maximum(bboxMap[:, :, 0], bboxMap[:, :, 1]) * np.maximum(bboxMap[:, :, 2], bboxMap[:, :, 3])))
                targetCls[bbox[1]:bbox[3], bbox[0]:bbox[2], gt_cls[j]] = 1
            Reg = np.vstack((Reg, targetReg.reshape(-1, 4)))
            Cls = np.vstack((Cls, targetCls.reshape(-1, self.clsNum)))
            Center = np.hstack((Center, targetCenter.reshape(-1)))
        return Cls, Reg, Center