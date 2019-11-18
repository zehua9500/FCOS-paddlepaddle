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

def val(modelpath):
    batch_size = 2
    print(time.asctime( time.localtime(time.time()) )[11:])
    def nms(data, threshold):
        #data type = numpy ,shape = N,6   [cls, score,x1, y1, x2, y2 ]
        #print("before ",data)
        scores = data[:,1]
        x1 = data[:,2]
        y1 = data[:,3]
        x2 = data[:,4]
        y2 = data[:,5]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        index = scores.argsort()[::-1]
        res = []
        while index.shape[0] >0:
            bbox = data[index[0]]
            res.append(bbox)
            x11 = np.maximum(bbox[2], x1[index[1:]])
            y11 = np.maximum(bbox[3], y1[index[1:]])
            x22 = np.minimum(bbox[4], x2[index[1:]])
            y22 = np.maximum(bbox[5], y2[index[1:]])
            
            w = np.maximum(0, x22 - x11 + 1) 
            h = np.maximum(0, y22 - y11 + 1)
            i = w*h
            iou = i / (areas[index[0]] + areas[index[1:]] - i)
            idx = iou < threshold
            index = index[1:][idx]
            
        #print("after ",np.array(res))   
        return res
    
    def valGenerator(valPath):
        testList = os.listdir(valPath)
        def __testImg__():
            for imgname in testList:
                img = cv2.imread(os.path.join(valPath, imgname))
                h,w = img.shape[:2]
                temp = np.zeros(shape = (640,640,3))
                temp[:h,:w,:] = img
                yield temp.transpose(2,0,1), imgname
        return  __testImg__
        
    def cal_map(labeldick, preddick, clsNum = 90):
        threshold = np.arange(0.5,1,0.05)
        totalNum = len(label)
        Map = [0] * threshold.size
        for imgname in labeldick:
            label = labeldick[imgname]
            pred = preddick[imgname]
            for i in range(threshold.size):
                Map[i] += fluid.layers.detection_map(fluid.dygraph.to_variable(pred), fluid.dygraph.to_variable(label), clsNum,overlap_threshold  = threshold[i],background_label = -1)
        return np.array(Map)/totalNum
        
    val_reader = valGenerator(r"work/coco/val2017")
    val_reader = paddle.batch(val_reader, batch_size= batch_size)
    
    start = time.time()
    with fluid.dygraph.guard():
        model = FCOS("fcos", 90, batch=batch_size, is_trainning=False)
        parameters, _ = fluid.dygraph.load_persistables(modelpath)
        model.load_dict(parameters)
        res = {}
        for valData in val_reader():
            img = fluid.dygraph.to_variable(np.stack([dat[0] for dat in valData], axis=0).astype(np.float32))
            imgname = [dat[1] for dat in valData]
            print(imgname)
            Cls, Scores, Centerness, Loc = model(img)
            for i in imgname:
                res[i] = np.zeros(shape = (0,6))  #[x1,y1,x2,y2, score, cls]
            feat_size = [[80,80], [40, 40], [20, 20], [10, 10], [5, 5]]
            stride = [8, 16, 32, 64, 128]
            threshold = 0.01
    
            for i in range(5):
                score = np.squeeze( (Scores[i] * Centerness[i]).numpy(), axis = 3 )
                #score = np.squeeze( Scores[i].numpy(), axis = 3 )
                idx = score > threshold
                cls = Cls[i].numpy()#[:,np.newaxis,:,:]
                reg = Loc[i].numpy()# * stride[i]
                for batch_idx in range(batch_size):
                    b_idx = idx[batch_idx]
                    b_cls = cls[batch_idx]
                    b_reg = reg[batch_idx]
                    bbox = np.zeros_like(b_reg)
                    bbox[:,:,0] = np.arange(feat_size[i][1]).reshape(1,-1) - b_reg[:,:,0]
                    bbox[:,:,1] = np.arange(feat_size[i][0]).reshape(-1,1) - b_reg[:,:,2]
                    bbox[:,:,2] = np.arange(feat_size[i][1]).reshape(1,-1) + b_reg[:,:,1]
                    bbox[:,:,3] = np.arange(feat_size[i][0]).reshape(-1,1) + b_reg[:,:,3]
                    
                    bbox = np.dstack([b_cls, score[batch_idx], bbox * stride[i]])
                    bbox = bbox[b_idx]
                    print(2)
                    print("bbox1 ",bbox.shape)
                    bbox = np.clip(bbox,0,640)
                    if(bbox.shape[0] == 0):continue
                    print("bbox1 ",bbox.shape)
                    bbox = nms(bbox, 0.5)
                    print("bbox2 ",len(bbox))
                    res[imgname[batch_idx]] = np.vstack((res[imgname[batch_idx]], bbox))
                    
                    
    with open(r"work/coco/annotations/instances_val2017.json", 'r') as f:
        js = json.load(f)
    label = {}
    for info in js["annotations"]:
        if not info['image_id'] in label:
            label[info['image_id']] = []
        bbox = info["bbox"]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        label[info['image_id']].append([info['category_id'] - 1] + bbox)    
    
    for imgname in label:
        if imgname not in res:
            res[imgname] = np.zeros(shape = (0,6))
    
    Map = cal_map(label, res, clsNum = 90)   
    print("map is ", np.mean(Map))
    print(Map)
    print(time.time() - start)
    del model