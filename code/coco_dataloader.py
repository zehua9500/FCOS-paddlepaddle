import numpy as np
import os
import cv2
import json
import random
import time
import math
import collections
import copy
from coco_encode import CocoEncode


class Coco_datGenerator():
    def __init__(self, batchsize=1):
        with open(r"work/coco/annotations/instances_train2017.json", 'r') as f:
            self.js = json.load(f)
        self.label = self.js2label()
        self.id2info = self.js2imgid()
        self.encode = CocoEncode()
        self.batchsize = batchsize
        self.mirror = 0.4
        self.flip = 0.4
        self.mixup = 0.35
        self.mixup_defect = 0.4
        self.length = [0, 6400, 8000, 8400, 8500, 8525]
        self.area = [self.length[i + 1] - self.length[i] for i in range(5)]

    def js2imgid(self):
        imgid = {}
        for info in self.js["images"]:
            temp = {}
            temp["imgname"] = info['file_name']
            temp["w"] = info['width']
            temp["h"] = info['height']
            imgid[info["id"]] = temp
        return imgid

    def js2label(self):
        label = {}
        for info in self.js["annotations"]:
            if not info['image_id'] in label:
                label[info['image_id']] = []
            bbox = info["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            label[info['image_id']].append(bbox + [info['category_id'] - 1])
        return label

    """
    def defect_aug(self, img, anno):
        if(np.sum(img > 220) > 244600):return img, anno
        if(anno.shape[0] == 0):return img, anno
        _temp = np.zeros_like(img)

        for i in range(anno.shape[0]):
            bbox = anno[i]
            _temp[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            
        #_nums = anno.shape[0] if anno.shape[0]<=4 else 4
        for i in range(anno.shape[0]):
            bbox = anno[i]
            h,w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if(h>=997):index_h = 0
            else:index_h = random.randint(0, 999 - h)
            if(w>=2443):index_w = 0
            else:index_w = random.randint(0, 2445 - w)
            
            if(np.sum(_temp[index_h:index_h+h, index_w:index_w+w]) > 0.1*h*w):continue
            _temp[index_h:index_h+h, index_w:index_w+w] = 1
            img[index_h:index_h+h, index_w:index_w+w] = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            new_bbox = np.array([index_w, index_h, index_w+w, index_h+h, bbox[4]])
            anno = np.vstack((anno, new_bbox))
        return img, anno
            
   """

    def dataAug(self, Img, anno, w, h):
        # if(False):
        if (random.random() < self.mirror):
            Img = np.flip(Img, 1)
            anno_w = anno[:, 2] - anno[:, 0]
            anno[:, 2] = w - anno[:, 0]
            anno[:, 0] = anno[:, 2] - anno_w

        if (random.random() < self.flip):
            Img = np.flip(Img, 0)
            anno_h = anno[:, 3] - anno[:, 1]
            anno[:, 3] = h - anno[:, 1]
            anno[:, 1] = anno[:, 3] - anno_h
        return Img, anno

    def generator(self):
        batchsize = self.batchsize
        length = len(self.label) // batchsize
        imgfold = r"work/coco/train2017"
        for imgid in self.label:
            imgbatch = np.zeros(shape=(batchsize, 3, 640, 640))
            clsbatch = np.zeros(shape=(batchsize, 8525, 90))  # 90类
            regbatch = np.zeros(shape=(batchsize, 8525, 4))
            cenbatch = np.zeros(shape=(batchsize, 8525))
            cenMaskbatch = np.zeros(shape=(batchsize, 8525))
            # w_max = 0
            # h_max = 0
            for idx in range(batchsize):
                bbox = np.array(self.label[imgid])
                imginfo = self.id2info[imgid]
                img = cv2.imread(os.path.join(imgfold, imginfo['imgname']))
                img, bbox = self.dataAug(img, bbox, imginfo['w'], imginfo['h'])
                # if(imginfo['w'] > w_max):w_max = imginfo['w']
                # if(imginfo['h'] > h_max):h_max = imginfo['h']
                Cls, Reg, Center = self.encode.encode(bbox)
                imgbatch[idx, :, :imginfo['h'], :imginfo['w']] = img.transpose(2, 0, 1)
                clsbatch[idx] = Cls
                regbatch[idx] = Reg
                cenbatch[idx] = Center
                cenMaskbatch[idx] = (Center > 0)

            mask = np.zeros_like(cenMaskbatch)
            for i in range(5):
                sub_cen = cenMaskbatch[:, self.length[i]:self.length[i + 1]]
                pos_num = np.sum(sub_cen)
                if (pos_num == 0):
                    mask[:, self.length[i]:self.length[i + 1]] = 0.05
                else:
                    mask[:, self.length[i]:self.length[i + 1]] = sub_cen * (self.area[i] - pos_num) / (pos_num * 10) + 1
            yield imgbatch.astype(np.float32), clsbatch.astype(np.float32), regbatch.astype(np.float32), cenbatch.astype(np.float32), cenMaskbatch.astype(
                np.float32), mask.astype(np.float32)
            # yield x.astype(np.float32)[np.newaxis,], Cls.astype(np.float32)[np.newaxis,], Reg.astype(np.float32)[np.newaxis,], Center.astype(np.float32)[np.newaxis,], Cen_Mask.astype(np.float32)[np.newaxis,]
