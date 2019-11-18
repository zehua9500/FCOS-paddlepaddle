import numpy as np
import os
import cv2
import json
import random
import time
import math
import collections
import copy
from Encode import Encode


class datGenerator():
    def __init__(self):
        self.category_id = {"破洞": 1, "水渍": 2, "油渍": 2, "污渍": 2,
                            "三丝": 3, "结头": 4, "花板跳": 5, "百脚": 6,
                            "毛粒": 7, "粗经": 8, "松经": 9, "断经": 10,
                            "吊经": 11, "粗维": 12, "纬缩": 13, "浆斑": 14,
                            "整经结": 15, "星跳": 16, "跳花": 16, "断氨纶": 17,
                            "稀密档": 18, "浪纹档": 18, "色差档": 18, "磨痕": 19,
                            "轧痕": 19, "修痕": 19, "烧毛痕": 19, "死皱": 20,
                            "云织": 20, "双纬": 20, "双经": 20, "跳纱": 20,
                            "筘路": 20, "纬纱不良": 20}
        self.category_id = {key: self.category_id[key] - 1 for key in self.category_id}
        self.img_h = 1000 - 1
        self.img_w = 2446 - 1
        self.clsType = [i for i in self.category_id]
        with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/label.json", 'r') as f:
            self.js = json.load(f)
        self.defect_path = r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"
        self.normal_path = r"work/train_dat/guangdong1_round1_train1_20190818/normal_Images"
        self.normal_list = os.listdir(self.normal_path)
        self.defect_list = os.listdir(self.defect_path)
        self.label = self.js2label()
        self.Buffer = self.bufferGenerater(size=5)
        self.encode = Encode()
        self.mirror = 0.4
        self.flip = 0.4
        self.mixup = 0.35
        self.mixup_defect = 0.4
        self.id2category = {0: "破洞", 1: random.choice(["水渍", "油渍", "污渍"]),
                            2: "三丝", 3: "结头", 4: "花板跳",
                            5: "百脚", 6: "毛粒", 7: "粗经",
                            8: "松经", 9: "断经",
                            10: "吊经", 11: "粗维", 12: "纬缩", 13: "浆斑",
                            14: "整经结", 15: random.choice(["星跳", "跳花"]),
                            16: "断氨纶", 17: random.choice(["稀密档", "浪纹档", "色差档"]),
                            18: random.choice(["磨痕", "轧痕", "修痕", "烧毛痕"]),
                            19: random.choice(["死皱", "云织", "双纬", "双经", "跳纱", "筘路", "纬纱不良"])}
        self.length = [0, 38250, 47889, 50353, 50977, 51137]
        self.area = [self.length[i + 1] - self.length[i] for i in range(5)]

    def bufferGenerater(self, size=3):
        buffer = {i: collections.deque(maxlen=size) for i in self.category_id}  # buffer中为【x1,y1,x2,y2,w,h】
        total = len(buffer) * size
        count = 0
        for info in self.js:
            defect_name = info["defect_name"]
            if (len(buffer[defect_name]) < size):
                count += 1
                img = cv2.imread(os.path.join(self.defect_path, info["name"]))
                bbox = [int(i) for i in info["bbox"]]
                dat = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                buffer[defect_name].append(dat)
            if count >= total: break
        return buffer

    def js2label(self):
        label = {}
        for info in self.js:
            if not info["name"] in label:
                label[info["name"]] = []
            label[info["name"]].append(info["bbox"] + [self.category_id[info["defect_name"]]])

        for key in label:
            temp = np.array(label[key])  # 将bbox转为numpy类型,并去小数
            temp[:, [2, 3]] = np.ceil(temp[:, [2, 3]])
            temp[:, 2] = np.clip(temp[:, 2], 0, 2445)
            temp[:, 3] = np.clip(temp[:, 3], 0, 999)
            temp[:, [0, 1]] = np.clip(temp[:, [0, 1]], 0, 2445)
            temp[:, [0, 1]] = np.floor(temp[:, [0, 1]])
            label[key] = temp.astype(np.int32)

        for imgname in self.normal_list:
            label[imgname] = None
        return label

    def defect_aug(self, img, anno):
        if (np.sum(img > 220) > 100000): return img, anno
        if (anno.shape[0] == 0): return img, anno
        _temp = np.zeros_like(img)

        for i in range(anno.shape[0]):
            bbox = anno[i]
            _temp[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        # _nums = anno.shape[0] if anno.shape[0]<=4 else 4
        for i in range(anno.shape[0]):
            bbox = anno[i]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h >= 997):
                index_h = 0
            else:
                index_h = random.randint(0, 999 - h)
            if (w >= 2443):
                index_w = 0
            else:
                index_w = random.randint(0, 2445 - w)

            if (np.sum(_temp[index_h:index_h + h, index_w:index_w + w]) > 0.1 * h * w): continue
            _temp[index_h:index_h + h, index_w:index_w + w] = 1
            img[index_h:index_h + h, index_w:index_w + w] = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            new_bbox = np.array([index_w, index_h, index_w + w, index_h + h, bbox[4]])
            anno = np.vstack((anno, new_bbox))
        return img, anno

    def dataAug(self, Img, anno):
        # if(False):
        if (random.random() < self.mirror):
            Img = np.flip(Img, 1)
            # Img = Img[:,::-1,:]
            # if (anno.shape[0] > 0):
            anno_w = anno[:, 2] - anno[:, 0]
            anno[:, 2] = 2446 - anno[:, 0]
            anno[:, 0] = anno[:, 2] - anno_w

        # if(False):
        if (random.random() < self.flip):
            Img = np.flip(Img, 0)
            # Img = Img[::-1,:,:]
            # if (anno.shape[0] > 0):
            anno_h = anno[:, 3] - anno[:, 1]
            anno[:, 3] = 1000 - anno[:, 1]
            anno[:, 1] = anno[:, 3] - anno_h
        return Img, anno

    def normalImgAddSample(self, img):
        if (random.random() < 0.35): return img, np.zeros((0, 5))
        temp = np.zeros_like(img, dtype=np.int32)
        highlight = (img > 220)
        img_mean = np.mean(img, axis=(0, 1))
        annos = []
        for i in range(5):
            add_type = random.choice(self.clsType)
            dif = []
            for buf_img in self.Buffer[add_type]:
                mean = np.mean(np.array(buf_img), axis=(0, 1))
                dif.append(np.sum((img_mean - mean) ** 2))
            index = np.argmin(dif)
            add_img = self.Buffer[add_type][index]
            [h, w] = add_img.shape[:2]

            if (h >= self.img_h):
                index_h = 0
            else:
                index_h = random.randint(0, self.img_h - h)

            if (w >= self.img_w):
                index_w = 0
            else:
                index_w = random.randint(0, self.img_w - w)

            if (np.sum(temp[index_h:index_h + h, index_w:index_w + w]) == 0 and \
                    np.sum(highlight[index_h:index_h + h, index_w:index_w + w]) <= 0.1 * h * w):
                cover_area_mean = np.mean(img[index_h:index_h + h, index_w:index_w + w], axis=(0, 1))
                # if(np.mean(cover_area_mean) > 170):continue
                temp[index_h:index_h + h, index_w:index_w + w] = add_img * (cover_area_mean / np.mean(add_img, axis=(0, 1)))
                bbox = [index_w, index_h, w, h, self.category_id[add_type]]
                annos.append(bbox)

        if (len(annos) == 0): return img, np.zeros((0, 5))
        annos = np.array(annos)
        annos[:, 2] += annos[:, 0]
        annos[:, 3] += annos[:, 1]
        img = img * (temp == 0) + temp
        return img, annos

    def generator(self):
        # random.shuffle(self.normal_list)
        # imgnames = self.normal_list[:800] + self.defect_list
        random.shuffle(self.defect_list)
        print("img nums = ", len(self.defect_list))
        for name in self.defect_list:
            if self.label[name] is None:
                x = cv2.imread(os.path.join(self.normal_path, name))
                x, y = self.normalImgAddSample(x)
                y = np.zeros((0, 5))
            else:
                x = cv2.imread(os.path.join(self.defect_path, name))
                y = copy.copy(self.label[name])

                if (random.random() < 0.05):
                    index = y.shape[0] % 3 - 1
                    cls = self.id2category[y[index][4]]
                    bbox = y[index]
                    self.Buffer[cls].append(x[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            if (random.random() < self.mixup):
                tempImg = cv2.imread(os.path.join(self.normal_path, random.choice(self.normal_list)))
                if (np.sum(tempImg > 220) < 100000):
                    x = (x * 0.95 + tempImg * 0.05).astype(np.uint8)
            if (random.random() < self.mixup_defect):
                x, y = self.defect_aug(x, y)
            if (random.random() < 0.2):
                x, y = self.defect_aug(x, y)
            x, y = self.dataAug(x, y)
            Cls, Reg, Center, divide_gt, roi_label = self.encode.encode(y)
            x = x.transpose(2, 0, 1)
            Cen_Mask = (Center > 0)
            Cen_Mask = Cen_Mask.astype(np.float32)[np.newaxis,]
            mask = np.zeros_like(Cen_Mask)
            for i in range(5):
                sub_cen = Cen_Mask[:, self.length[i]:self.length[i + 1]]
                pos_num = np.sum(sub_cen)
                if (pos_num == 0):
                    mask[:, self.length[i]:self.length[i + 1]] = 0.01
                else:
                    mask[:, self.length[i]:self.length[i + 1]] = sub_cen * np.min((np.max(((self.area[i] - pos_num) / (pos_num * 20), 1)), 1000)) + (
                                1 + i * 0.05)
            yield x.astype(np.float32)[np.newaxis,], \
                  Cls.astype(np.float32)[np.newaxis,], \
                  Reg.astype(np.float32)[np.newaxis,], \
                  Center.astype(np.float32)[np.newaxis,], \
                  Cen_Mask, \
                  mask.astype(np.float32), \
                  divide_gt(np.float32), \
                  roi_label
