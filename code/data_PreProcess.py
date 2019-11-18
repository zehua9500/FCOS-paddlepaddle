import cv2
import numpy as np
import json
import os
import random
from matplotlib import pyplot as plt
import copy
import collections

with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train.json", 'r') as f:
    js_old = json.load(f)

with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train2.json", 'r') as f:
    js_new = json.load(f)

defect_path = r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"
normal_path = r"work/train_dat/guangdong1_round1_train1_20190818/normal_Images"
js = js_old + js_new

category2id = {"破洞": 1, "水渍": 2, "油渍": 2, "污渍": 2,
               "三丝": 3, "结头": 4, "花板跳": 5, "百脚": 6,
               "毛粒": 7, "粗经": 8, "松经": 9, "断经": 10,
               "吊经": 11, "粗维": 12, "纬缩": 13, "浆斑": 14,
               "整经结": 15, "星跳": 16, "跳花": 16, "断氨纶": 17,
               "稀密档": 18, "浪纹档": 18, "色差档": 18, "磨痕": 19,
               "轧痕": 19, "修痕": 19, "烧毛痕": 19, "死皱": 20,
               "云织": 20, "双纬": 20, "双经": 20, "跳纱": 20,
               "筘路": 20, "纬纱不良": 20}

id2category = {1: "破洞", 2: random.choice(["水渍", "油渍", "污渍"]),
               3: "三丝", 4: "结头", 5: "花板跳",
               6: "百脚", 7: "毛粒", 8: "粗经",
               9: "松经", 10: "断经",
               11: "吊经", 12: "粗维", 13: "纬缩", 14: "浆斑",
               15: "整经结", 16: random.choice(["星跳", "跳花"]),
               17: "断氨纶", 18: random.choice(["稀密档", "浪纹档", "色差档"]),
               19: random.choice(["磨痕", "轧痕", "修痕", "烧毛痕"]),
               20: random.choice(["死皱", "云织", "双纬", "双经", "跳纱", "筘路", "纬纱不良"])}
img_label = {}
for info in js:
    if info["name"] not in img_label:
        img_label[info["name"]] = []
    img_label[info["name"]].append(info["bbox"] + [category2id[info["defect_name"]]])
"""
defect_w = []
defect_h = []
#area = []
for info in js:
    bbox = info['bbox']
    defect_w.append(bbox[2] - bbox[0])
    defect_h.append(bbox[3] - bbox[1])
    #area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) 
plt.title("w_h") 
plt.xlabel("w") 
plt.ylabel("h") 
plt.plot(defect_w,defect_h,".r") 
plt.show()
"""

cls_num = [0] * 20
cls_type = [""] * 20  # [i for i in range(20)]
for cls in category2id:
    index = category2id[cls]
    cls_type[index - 1] += cls + " "
cls_type[19] = cls_type[19][:10]
for info in js:
    idx = category2id[info["defect_name"]] - 1
    cls_num[idx] += 1
plt.rc('font', family='SimHei', size=13)
"""
width = 1
idx = np.arange(len(cls_type))
plt.bar(idx,cls_num,width, align =  'center') 
plt.title('cls_num') 
plt.ylabel('Nums') 
plt.xlabel('Cls') 
plt.xticks(idx+width/2, cls_type, rotation=90)
plt.show()
"""
mean_area = [[] for i in range(20)]
for info in js:
    cls = info["defect_name"]
    bbox = info["bbox"]
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    mean_area[category2id[cls] - 1].append(area)
mean_area = [np.mean(area) for area in mean_area]
mean_area = np.sqrt(mean_area)
"""
width = 1
idx = np.arange(20)
plt.bar(idx,mean_area,width, align =  'center') 
plt.title('cls_area') 
plt.ylabel('area') 
plt.xlabel('Cls') 
plt.xticks(idx+width/2, cls_type, rotation=90)
plt.show()
"""
pix_num = ((mean_area < 160) * (mean_area / 8) * cls_num) + \
          ((mean_area < 320) * (mean_area > 80) * (mean_area / 16) * cls_num) + \
          ((mean_area < 500) * (mean_area > 160) * (mean_area / 32) * cls_num) + \
          ((mean_area > 320) * (mean_area / 64) * cls_num) + \
          ((mean_area > 500) * (mean_area / 128) * cls_num)
add_num = (np.max(pix_num) / pix_num) * ((np.array(mean_area) < 50) + 1)
add_num = np.round(add_num * 0.6).astype(np.int32)

add_img_label = {}
target_path = r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"
src_path = r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"
norm_path = r"work/train_dat/guangdong1_round1_train1_20190818/normal_Images"
norm_list = os.listdir(norm_path)

test_label = copy.deepcopy(img_label)
print("total = ", len(test_label))
count = collections.deque(maxlen=300)

for idx, imgname in enumerate(test_label):
    bbox = np.array(test_label[imgname])
    bbox = np.round(bbox).astype(np.int32)
    cls = bbox[:, -1]
    add_times = []
    for i in range(cls.shape[0]):
        add_times.append(add_num[cls[i] - 1])
    add_times = int(np.mean(add_times))
    if (add_times == 1): continue
    img = cv2.imread(os.path.join(src_path, imgname))
    count.append(add_times)
    for i in range(1, add_times):
        new_img_name = imgname[:-4] + "_" + str(i) + ".jpg"
        new_img = copy.deepcopy(img)
        new_bbox = copy.deepcopy(bbox)
        cut_h = random.randint(0, 100)
        cut_w = random.randint(0, 200)
        new_img = new_img[cut_h:, cut_w:]
        new_bbox[:, [0, 2]] = new_bbox[:, [0, 2]] - cut_w
        new_bbox[:, [1, 3]] = new_bbox[:, [1, 3]] - cut_h
        if (np.sum(new_bbox < 0) > 0): continue
        scale = np.array([1000, 2446]) / new_img.shape[:2]
        new_img = cv2.resize(new_img, (2446, 1000))
        new_bbox[:, [0, 2]] = new_bbox[:, [0, 2]] * scale[1]
        new_bbox[:, [1, 3]] = new_bbox[:, [1, 3]] * scale[0]
        while (True):
            norm_add_img = cv2.imread(os.path.join(norm_path, random.choice(norm_list)))
            if (np.sum(norm_add_img > 200) < 104600):
                break
        new_img_mean = np.mean(new_img, axis=(0, 1))
        add_img_mean = np.mean(norm_add_img, axis=(0, 1))
        new_img = (new_img * 0.85 + norm_add_img * 0.15 * (new_img_mean / add_img_mean)).astype(np.uint8) #原为0.8 ： 0.2
        add_img_label[new_img_name] = new_bbox
        cv2.imwrite(os.path.join(target_path, new_img_name), new_img)
    if (idx % 200 == 0):
        print("idx = %d, mean_times = %d" % (idx, np.mean(count)))

add_js = []
for imgname in add_img_label:
    anno = add_img_label[imgname].astype(np.float)
    for i in range(anno.shape[0]):
        temp = {}
        temp["bbox"] = list(anno[i])[:4]
        temp["name"] = imgname
        temp["defect_name"] = id2category[anno[i][-1]]
        add_js.append(temp)

with open(r"E:\guangdong\guangdong1_round1_train1_20190818\guangdong1_round1_train1_20190818\Annotations\anno_train_add.json",'w') as fp:
    json.dump(add_js, fp, indent=4, separators=(',', ': '))