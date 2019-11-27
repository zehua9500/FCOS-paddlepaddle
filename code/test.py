import numpy as np
import os
import pandas as pd
import paddle.fluid as fluid
import paddle
import cv2
import json
import random
import time
import collections
import math
import sys

sys.path.append(r"work/fcos")
from FCOS import FCOS


def test(paramPath=r"work/Model/model1/epochs=17",
         save_path=r"'work/res/result17_centerness.json'",
         test_fold_path=r"work/test_data/guangdong1_round1_testA_20190818"):
    batch_size = 2

    def nms(data, threshold):
        scores = data[:, 4]
        x1 = data[:, 0]
        y1 = data[:, 1]
        x2 = data[:, 2]
        y2 = data[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        index = scores.argsort()[::-1]
        res = []
        while index.shape[0] > 0:
            bbox = data[index[0]]
            res.append(bbox)
            x11 = np.maximum(bbox[0], x1[index[1:]])
            y11 = np.maximum(bbox[1], y1[index[1:]])
            x22 = np.minimum(bbox[2], x2[index[1:]])
            y22 = np.maximum(bbox[3], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            i = w * h
            iou = i / (areas[index[0]] + areas[index[1:]] - i)
            idx = iou < threshold
            index = index[1:][idx]
        return res

    def testGenerator(testPath):
        testList = os.listdir(testPath)

        def __testImg__():
            for imgname in testList:
                img = cv2.imread(os.path.join(testPath, imgname))
                yield img.transpose(2, 0, 1), imgname

        return __testImg__

    test_reader = testGenerator(test_fold_path)
    test_reader = paddle.batch(test_reader, batch_size=2)

    start = time.time()
    with fluid.dygraph.guard():
        model = FCOS("fcos", 20, batch=batch_size, is_trainning=False)
        parameters, _ = fluid.dygraph.load_persistables(paramPath)
        model.load_dict(parameters)
        model.eval()
        res = {}
        for testData in test_reader():
            img = fluid.dygraph.to_variable(np.stack([dat[0] for dat in testData], axis=0).astype(np.float32))
            imgname = [dat[1] for dat in testData]
            Cls, Scores, Centerness, Loc = model(img)
            for i in imgname:
                res[i] = np.zeros(shape=(0, 6))  # [x1,y1,x2,y2, score, cls]
            feat_size = [(125, 306), (63, 153), (32, 77), (16, 39), (8, 20)]
            stride = [8, 16, 32, 64, 128]
            threshold = 0.45

            for i in range(5):
                score = np.squeeze((Scores[i] * Centerness[i]).numpy(), axis=3)
                # score = np.squeeze( Scores[i].numpy(), axis = 3 )
                idx = score > threshold
                cls = Cls[i].numpy()  # [:,np.newaxis,:,:]
                reg = Loc[i].numpy()  # * stride[i]
                for batch_idx in range(batch_size):
                    b_idx = idx[batch_idx]
                    b_cls = cls[batch_idx]
                    b_reg = reg[batch_idx]
                    bbox = np.zeros_like(b_reg)
                    bbox[:, :, 0] = np.arange(feat_size[i][1]).reshape(1, -1) - b_reg[:, :, 0]
                    bbox[:, :, 1] = np.arange(feat_size[i][0]).reshape(-1, 1) - b_reg[:, :, 2]
                    bbox[:, :, 2] = np.arange(feat_size[i][1]).reshape(1, -1) + b_reg[:, :, 1]
                    bbox[:, :, 3] = np.arange(feat_size[i][0]).reshape(-1, 1) + b_reg[:, :, 3]
                    bbox = np.dstack([bbox * stride[i], score[batch_idx], b_cls])
                    bbox = bbox[b_idx]
                    res[imgname[batch_idx]] = np.vstack((res[imgname[batch_idx]], bbox))

        ans = []
        for key in res:
            if (res[key].shape[0] > 0):
                bbox = nms(res[key], 0.2)
                for i in range(len(bbox)):
                    temp = {}
                    temp["name"] = key
                    temp["category"] = int(bbox[i][5]) + 1
                    temp["bbox"] = list(bbox[i][:4])
                    temp["score"] = bbox[i][4]
                    ans.append(temp)

        with open(save_path, 'w') as fp:
            json.dump(ans, fp, indent=4, separators=(',', ': '))
    print(time.time() - start)
