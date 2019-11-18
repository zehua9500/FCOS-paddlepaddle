#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[4]:


#!unzip data/data11713/train1_20190818.zip -d work/train_dat/
#!unzip data/data11713/guangdong1_round1_testA_20190818.zip -d work/test_data/
#!zip -r work/train_data.zip work/train_dat/guangdong1_round1_train1_20190818
#!unzip data/data7122/test2017.zip -d work/coco


# In[8]:


import numpy as np
a = np.arange(4).reshape(2,2)
b = np.array([[True,True],[False,True]])
c = a[b]
d = np.array([1,2,3])
np.hstack([c,d])


# In[2]:


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
with fluid.dygraph.guard():
    a = np.array([1,2,3,45])
    a = fluid.dygraph.to_variable(a)
    print(a>0)


# In[1]:


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

"""
from coco_dataloader import Coco_datGenerator
from coco_valdataloader import val
batch_size = 4
train_generator = Coco_datGenerator(batchsize = batch_size)
train_reader = train_generator.generator
img = fluid.layers.data(name='img', dtype='float32', shape=[3,640,640])
cls = fluid.layers.data(name='cls', dtype='float32', shape=[8525, 90])
cen = fluid.layers.data(name='cen', dtype='float32', shape=[8525])
reg = fluid.layers.data(name='reg', dtype='float32', shape=[8525, 4])
cen_mask = fluid.layers.data(name='cen_mask', dtype='float32', shape=[8525])
py_reader = fluid.io.PyReader(feed_list=[img, cls, reg, cen, cen_mask], capacity=2, iterable=True)
py_reader.decorate_batch_generator(train_reader, places=fluid.cuda_places(0))
"""

batch_size = 1
from data import datGenerator
train_generator = datGenerator()
train_reader = train_generator.generator
img = fluid.layers.data(name='img', dtype='float32', shape=[3,1000,2446])
cls = fluid.layers.data(name='cls', dtype='float32', shape=[51137, 20])
cen = fluid.layers.data(name='cen', dtype='float32', shape=[51137])
reg = fluid.layers.data(name='reg', dtype='float32', shape=[51137, 4])
cen_mask = fluid.layers.data(name='cen_mask', dtype='float32', shape=[51137])

py_reader = fluid.io.PyReader(feed_list=[img, cls, reg, cen, cen_mask], capacity=2, iterable=True)
py_reader.decorate_batch_generator(train_reader, places=fluid.cuda_places(0))
#py_reader.decorate_batch_generator(train_reader, places=fluid.cpu_places(2))


# In[2]:


with fluid.dygraph.guard():
    print(data[2].shape)


# In[3]:


epochs = 5
learning_rate = 5e-6
start = time.time()

#cls_loss_hist = collections.deque(maxlen=300)
#reg_loss_hist = collections.deque(maxlen=300)
#cen_loss_hist = collections.deque(maxlen=300)
cls_loss_hist = []
reg_loss_hist = []
cen_loss_hist = []
dice_loss_hist = []
learning_rate = fluid.layers.cosine_decay( learning_rate = learning_rate, step_each_epoch=12872//10, epochs=60)
adam = fluid.optimizer.AdamOptimizer(learning_rate = learning_rate)

with fluid.dygraph.guard():
    model = FCOS("fcos", 20, batch=batch_size)
    parameters, _ = fluid.dygraph.load_persistables(r"work/model/v4/7head_eps = 10_half")
    model.load_dict(parameters)
    print("training begin.....")
    for epoch in range(10, epochs+10):
        for idx, data in enumerate(py_reader()):
            #if(data[2].shape[0] > batch_size):
            #    print("batch error  ")
            #    break
            data[4].stop_gradient = True
            data[1].stop_gradient = True
            data[2].stop_gradient = True
            data[3].stop_gradient = True
            data[5].stop_gradient = True
            data[6].stop_gradient = True
            data[7].stop_gradient = True
            f_loss, i_loss, c_loss = model(data[0], data[1], data[2], data[3],data[4],data[5],data[6],data[7])
            loss = f_loss + i_loss + c_loss# + d_loss * fluid.layers.reduce_sum(data[4]) * 0.1 / batch_size
            loss.backward()
            adam.minimize(loss)
            cls_loss_hist.append(f_loss.numpy())
            reg_loss_hist.append(i_loss.numpy())
            cen_loss_hist.append(c_loss.numpy())
            #dice_loss_hist.append(d_loss.numpy())
            if(idx % 300 == 0):
                cls_loss_mean = np.mean(cls_loss_hist)
                reg_loss_mean = np.mean(reg_loss_hist)
                cen_loss_mean = np.mean(cen_loss_hist)
                #dice_loss_mean = np.mean(dice_loss_hist)
                cls_loss_hist = []
                reg_loss_hist = []
                cen_loss_hist = []
                #dice_loss_hist = []
                loss_mean = cls_loss_mean + reg_loss_mean + cen_loss_mean# + dice_loss_mean
                #print(time.asctime( time.localtime(time.time()) )[11:])
                print("epoch = %d | iter = %d | loss = %.5f | focal_loss = %.5f | iou_loss = %.5f | centerness_loss = %.5f | use time = %.3f s | time is %s"%                (epoch, idx, loss_mean, cls_loss_mean, reg_loss_mean, cen_loss_mean, time.time() - start, time.asctime(time.localtime(time.time()) )[11:]))
                start = time.time()
            model.clear_gradients()
        fluid.dygraph.save_persistables(model.state_dict(), "work/model/v4/7head_eps = %s"%epoch)


# In[4]:


fluid.dygraph.save_persistables(model.state_dict(), "work/model/v4/7head_eps = 10_half")


# In[2]:


class Bottleneck(fluid.dygraph.Layer):
    def __init__(self, name_scope, planes, is_test, stride = 1, downsample=None):
        super(Bottleneck, self).__init__(name_scope)
        self.conv1 = fluid.dygraph.Conv2D("conv1", planes, 1)
        self.bn1 = fluid.dygraph.BatchNorm("bn1",planes, act = "relu", is_test = is_test)
        
        self.conv2 = fluid.dygraph.Conv2D("conv2", planes, 3, padding = 1, stride=stride)
        self.bn2 = fluid.dygraph.BatchNorm("bn2",planes, act = "relu", is_test = is_test)
        
        self.conv3 = fluid.dygraph.Conv2D("conv3", planes * 4, 1)
        self.bn3 = fluid.dygraph.BatchNorm("bn3",planes * 4, act = "relu", is_test = is_test)
        
        #self.relu = fluid.layers.relu
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        return self.bn3(x)


# In[3]:


class Make_layer(fluid.dygraph.Layer):
    def __init__(self, name_scope, planes, layernums, is_test, stride = 1):
        super(Make_layer, self).__init__(name_scope)
        self.layernums = layernums
        
        self.downsample = fluid.dygraph.Conv2D("downsample", planes * 4, 1, stride = stride)
        self.layer1 = Bottleneck("layer1", planes, stride = stride, is_test = is_test, downsample = self.downsample)
        
        self.layer2 = Bottleneck("layer2", planes, is_test = is_test)
        self.layer3 = Bottleneck("layer3", planes, is_test = is_test)
        
        if(layernums >= 4):
            self.layer4 = Bottleneck("layer4", planes, is_test = is_test)
            
        if(layernums >= 6):
            self.layer5 = Bottleneck("layer5", planes, is_test = is_test)
            self.layer6 = Bottleneck("layer6", planes, is_test = is_test)
            
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if(self.layernums >= 4):
            x = self.layer4(x)
            
        if(self.layernums >= 6):
            x = self.layer5(x)
            x = self.layer6(x)
        return x


# In[4]:


class SEConcat(fluid.dygraph.Layer):
    def __init__(self, name_scope, channel = 256):
        super(SEConcat, self).__init__(name_scope)
        self.downChannel = fluid.dygraph.Conv2D("downChannel", 256,filter_size = 1, stride=1)
        self.fc = fluid.dygraph.FC("fc", size = 256, act="sigmoid", is_test=False, dtype='float32')

    def forward(self, x1, x2):
        x = fluid.layers.concat(input=[x1, x2], axis=1)
        x = self.downChannel(x)
        mean = fluid.layers.reduce_mean(x,dim = [2,3])
        mean = self.fc(mean)
        return fluid.layers.elementwise_mul(x, mean, axis=0)
        


# In[5]:


class FPN(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test, channel = 256):
        super(FPN, self).__init__(name_scope)
        self.P5_1 = fluid.dygraph.Conv2D("P5_1", channel, filter_size = 3, stride=1, padding=1)
        self.P5_2 = fluid.dygraph.Conv2D("P5_2", channel, filter_size = 3, stride=1, padding=1)
        self.P5_up = fluid.dygraph.Conv2D("P5_up", channel, filter_size = 3, stride=1, padding=1)
        self.se5to4 = SEConcat("se5to4")
        
        self.P4_1 = fluid.dygraph.Conv2D("P4_1", channel, filter_size = 3, stride=1, padding=1)
        self.P4_2 = fluid.dygraph.Conv2D("P4_2", channel, filter_size = 3, stride=1, padding=1)
        self.P4_up = fluid.dygraph.Conv2D("P4_up", channel, filter_size = 3, stride=1, padding=1)
        self.se4to3 = SEConcat("se4to3")
        
        self.P3_1 = fluid.dygraph.Conv2D("P3_1", channel, filter_size = 3, stride=1, padding=1)
        self.P3_2 = fluid.dygraph.Conv2D("P3_2", channel, filter_size = 3, stride=1, padding=1)
        
        self.P6_1 = fluid.dygraph.Conv2D("P6_1", channel, filter_size = 3, stride=2, padding=1)
        self.bn6 = fluid.dygraph.BatchNorm("bn6",channel, act = "relu", is_test = is_test)
        
        self.P7_1 = fluid.dygraph.Conv2D("P7_1", channel, filter_size = 3, stride=2, padding=1)
    def forward(self, inputs):
        C3,C4,C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsample = fluid.layers.resize_nearest(input = P5_x, scale=None, out_shape = (63, 153))
        P5_upsample = self.P5_up(P5_upsample)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        #P4_x = P4_x + P5_upsample
        P4_x = self.se5to4(P4_x, P5_upsample)
        P4_upsample = fluid.layers.resize_nearest(input = P4_x, scale=None, out_shape = (125, 306))
        P4_upsample = self.P4_up(P4_upsample)
        P4_x = self.P4_2(P4_x)
        
        P3_x = self.P3_1(C3)
        #P3_x = P3_x + P4_upsample
        P3_x = self.se4to3(P3_x, P4_upsample)
        P3_x = self.P3_2(P3_x)
        
        P6_x = self.P6_1(C5)
        P6_bn = self.bn6(P6_x)
        
        P7_x = self.P7_1(P6_bn)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]
    


# In[6]:


class ASPP(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test):
        super(ASPP, self).__init__(name_scope)
        self.dilate1 = fluid.dygraph.Conv2D("dilate1", num_filters = 64, filter_size = 3, stride=1, padding=1, dilation=1)
        self.dilate2 = fluid.dygraph.Conv2D("dilate1", num_filters = 64, filter_size = 3, stride=1, padding=3, dilation=3)
        self.dilate3 = fluid.dygraph.Conv2D("dilate1", num_filters = 64, filter_size = 3, stride=1, padding=4, dilation=4)
        self.dilate4 = fluid.dygraph.Conv2D("dilate1", num_filters = 64, filter_size = 3, stride=1, padding=6, dilation=6)
        self.bn1 = fluid.dygraph.BatchNorm("bn1",256, act = "relu", is_test = is_test)
        
        self.merge = fluid.dygraph.Conv2D("merge", num_filters = 256, filter_size = 1, stride=1)
        self.bn2 = fluid.dygraph.BatchNorm("bn2",256, act = "relu", is_test = is_test)
        
    def forward(self, inputs):    
        x1 = self.dilate1(inputs)
        x2 = self.dilate2(inputs)
        
        x3 = self.dilate3(inputs)
        x4 = self.dilate4(inputs)
        out = fluid.layers.concat(input=[x1,x2,x3,x4], axis=1)
        out = self.bn1(out)
        out = self.merge(out)
        return self.bn2(out)


# In[7]:


class Head(fluid.dygraph.Layer):
    def __init__(self, name_scope, clsNum, is_test):
        super(Head, self).__init__(name_scope)
        self.cls1 = ASPP("cls1", is_test = is_test)
        self.cls2 = ASPP("cls2", is_test = is_test)
        self.cls3 = ASPP("cls3", is_test = is_test)
        #self.cls4 = ASPP("cls4", is_test = is_test)
        
        self.loc1 = ASPP("loc1", is_test = is_test)
        self.loc2 = ASPP("loc2", is_test = is_test)
        self.loc3 = ASPP("loc3", is_test = is_test)
        #self.loc4 = ASPP("loc4", is_test = is_test)

        self.cls_out = fluid.dygraph.Conv2D("class", num_filters = clsNum, filter_size = 3, stride=1, padding=1, dilation=1)
        self.center_ness = fluid.dygraph.Conv2D("center_ness", num_filters = 1, filter_size = 3, stride=1, padding=1, dilation=1)
        self.regression = fluid.dygraph.Conv2D("regression", num_filters = 4, filter_size = 3, stride=1, padding=1, dilation=1)
        
    def forward(self, inputs):
        cls = self.cls1(inputs)
        cls = self.cls2(cls)
        cls = self.cls3(cls)
        #cls = self.cls4(cls)
        
        cls_out = self.cls_out(cls)
        center_ness = self.center_ness(cls)

        
        loc = self.loc1(inputs)
        loc = self.loc2(loc)
        loc = self.loc3(loc)
        #loc = self.loc4(loc)
        loc = self.regression(loc)
        
        return [fluid.layers.sigmoid(cls_out), 
                fluid.layers.sigmoid(center_ness), 
                fluid.layers.exp(loc)]


# In[8]:


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test = False):
        super(ResNet, self).__init__(name_scope)
        self.conv1 = fluid.dygraph.Conv2D("conv1", num_filters = 64, filter_size = 7, stride=2, padding=3, dilation=1)
        self.bn1 = fluid.dygraph.BatchNorm("bn1",64, act = "relu", is_test = is_test)
        self.maxPooling  = fluid.dygraph.Pool2D("maxpooling", pool_size=2, pool_stride = 2, pool_type='max')

        self.block1 = Make_layer("block1", 64, layernums = 3, stride = 1, is_test = is_test)
        self.block2 = Make_layer("block2", 128, layernums = 4, stride = 2, is_test = is_test)
        self.block3 = Make_layer("block3", 256, layernums = 6, stride = 2, is_test = is_test)
        self.block4 = Make_layer("block4", 512, layernums = 3, stride = 2, is_test = is_test) #最后一层的输出是带激活函数的
        self.fpn = FPN("FPN", is_test = is_test)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxPooling(x)
        x = self.block1(x)
        #return [x]
        C3 = self.block2(x)
        C4 = self.block3(C3)
        C5 = self.block4(C4)
        #return [C3, C4, C5]
        return self.fpn([C3, C4, C5])


# In[9]:


#with fluid.dygraph.guard():
class Loss(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Loss, self).__init__(name_scope)
        self.balance_weight = fluid.dygraph.to_variable(np.array([6, 5, 2, 1.5, 14,14, 11, 9, 6, 7,10, 3, 6.6, 8, 7, 10,  3, 6, 5, 6], 
                                                        dtype = np.float32)) / 10
        self.balance_weight.stop_gradient = True
        self.limit = fluid.dygraph.to_variable(np.array([0], dtype = np.float32))
        self.limit.stop_gradient = True
                                   
    def iou_loss(self, pred, label, reg_mask):
        #label shape = [b, -1, 4]  l,r,t,b
        i_h = fluid.layers.elementwise_min(pred[:,:,0], label[:,:,0]) + fluid.layers.elementwise_min(pred[:,:,1], label[:,:,1])
        i_w = fluid.layers.elementwise_min(pred[:,:,2], label[:,:,2]) + fluid.layers.elementwise_min(pred[:,:,3], label[:,:,3])
        i_area = fluid.layers.elementwise_mul(i_h, i_w)
        u_area = fluid.layers.elementwise_mul(pred[:,:,0] + pred[:,:,1],  pred[:,:,2] + pred[:,:,3]) +                 fluid.layers.elementwise_mul(label[:,:,0] + label[:,:,1],  label[:,:,2] + label[:,:,3])
        iou = i_area / (u_area - i_area + 1e-7)
        #mask = fluid.layers.greater_than(label, self.limit)
        #mask = paddle.fluid.layers.cast(mask, dtype = "float32")
        loss = (1 - iou) * reg_mask
        return fluid.layers.reduce_sum(loss) / (fluid.layers.reduce_sum(reg_mask) + 1e-5)
        
    def centerness_loss(self, pred, label, reg_mask):
        #label shape  = [b, -1]
        #采用 L1 loss
        diff = pred - label
        loss = fluid.layers.elementwise_mul(diff, diff)
        loss = fluid.layers.elementwise_mul(loss,  5*label + 0.5)
        #loss = fluid.layers.pow(diff, factor=4.0)
        #loss = paddle.fluid.layers.abs(pred - label)
        #loss = 5*reg_mask*loss + 0.5*(1-reg_mask)*loss
        #loss = paddle.fluid.layers.cast(loss, dtype = "float32")
        #loss = fluid.layers.clip(loss,1e-4, 1.0)
        #sum_loss = fluid.layers.reduce_sum(loss).numpy()
        #print("centerness_loss sum = %s, is_nan = %s"%(sum_loss, np.isnan(sum_loss)))
        return fluid.layers.reduce_mean(loss) * 10
                
    def focal_loss(self, pred, label):
        #cls loss shape = [b,-1, clsNum]
        label = (1 - label)*0.005 + label *0.99
        alpha = 0.95
        gamma = 3.0
        eps = 1e-7
        bce_loss = -1 * (alpha * label * fluid.layers.log(pred + eps)* self.balance_weight + (1 - alpha) * (1 - label) * fluid.layers.log(1 - pred + eps))
        focal_weight = label * fluid.layers.pow((1 - pred), gamma) + (1 - label)*fluid.layers.pow(pred, gamma)
        cls_loss = bce_loss * focal_weight 
        return fluid.layers.reduce_mean(cls_loss) * 1000  #focal_loss值太少，故把mean换成sum
        
    def forward(self, cls_out, cls_label, reg_out, reg_label, cent_out, cent_label, reg_mask):
        return self.focal_loss(cls_out, cls_label),                 self.iou_loss(reg_out, reg_label, reg_mask),                 self.centerness_loss(cent_out, cent_label, reg_mask)


# In[10]:


class FCOS(fluid.dygraph.Layer):
    def __init__(self, name_scope, clsNum, batch = 8, is_trainning = True):
        super(FCOS, self).__init__(name_scope)
        
        self.trainning = is_trainning
        self.resnet = ResNet("ResNet", is_test = not is_trainning)
        self.head1 = Head('Head1', clsNum, is_test = not is_trainning) #head1用于前3层 fpn
        self.head2 = Head('Head2', clsNum, is_test = not is_trainning)#head2用于前3层 fpn
        self.stride = [8, 16, 32, 64, 128]
        self.clsNum = clsNum
        self.batch = batch
        self.loss =Loss("loss")
        print("FCOS load final")
        
    def forward(self, x, cls_label = None, reg_label = None, cent_label = None, reg_mask = None):
        feature = self.resnet(x)
        #Cls = paddle.fluid.layers.zeros(shape = (self.batch, 0, self.clsNum), dtype = "float32")
        #Reg = paddle.fluid.layers.zeros(shape = (self.batch, 0, 4), dtype = "float32")
        #Center = paddle.fluid.layers.zeros(shape = (self.batch, 0), dtype = "float32")
        Cls, Reg, Center = [],[],[]
        if(self.trainning):
            for idx, feat in enumerate(feature):
                if(idx < 3):
                    cls_out, center_ness, loc = self.head1(feat)
                else:
                    cls_out, center_ness, loc = self.head2(feat)
                cls_out = fluid.layers.transpose(cls_out, perm=[0, 2, 3, 1])
                cls_out = fluid.layers.reshape(x=cls_out, shape=[self.batch, -1, self.clsNum], inplace=False)
                Cls.append(cls_out)
    
                center_ness = fluid.layers.transpose(center_ness, perm=[0, 2, 3, 1])
                center_ness = fluid.layers.reshape(x=center_ness, shape=[self.batch, -1], inplace=False)
                Center.append(center_ness)
    
                loc = fluid.layers.transpose(loc, perm=[0, 2, 3, 1])
                loc = fluid.layers.reshape(x=loc, shape=[self.batch, -1, 4], inplace=False)
                Reg.append(loc)
                
            Reg = fluid.layers.concat(input=Reg, axis=1)
            Center = fluid.layers.concat(input=Center, axis=1)
            Cls = fluid.layers.concat(input=Cls, axis=1)
            return self.loss(Cls, cls_label, Reg, reg_label, Center, cent_label, reg_mask)
        else:
            Scores = []
            for feat in feature:
                cls_out, center_ness, loc = self.head(feat)
                cls_out = fluid.layers.transpose(cls_out, perm=[0, 2, 3, 1])
                center_ness = fluid.layers.transpose(center_ness, perm=[0, 2, 3, 1])
                loc = fluid.layers.transpose(loc, perm=[0, 2, 3, 1])
                
                argmax = fluid.layers.argmax(cls_out, axis=3)
                score = fluid.layers.reduce_max(cls_out, dim=3, keep_dim = True)
                Scores.append(score)
                Cls.append(argmax)
                Reg.append(loc)
                Center.append(center_ness)
            return Cls, Scores, Center, Reg


# In[11]:


class Encode():
    def __init__(self):
        self.divide = [100, 200, 300, 500]
        self.stride = [8, 16, 32, 64, 128]
        self.size2layer = [[125, 306], [63, 153], [32, 77], [16, 39], [8, 20]]
        self.layerNum = 5
        self.clsNum = 20
        
    def gt_process(self, gt):
        if gt is None:
            return [np.zeros((0,5))]*5
        meanedge = np.mean((gt[:,2] - gt[:,0],  gt[:,3] - gt[:,1]), axis=0)
        """
        gt1 = gt[meanedge < self.divide[0]]
        gt2 = gt[(meanedge >= self.divide[0]) * (meanedge < self.divide[1])]
        gt3 = gt[(meanedge >= self.divide[1]) * (meanedge < self.divide[2])]
        gt4 = gt[(meanedge >= self.divide[2]) * (meanedge < self.divide[3])]
        gt5 = gt[meanedge >= self.divide[3]]
        """
        gt1 = gt
        gt2 = gt[meanedge > self.divide[0]]
        gt3 = gt[meanedge > self.divide[1]]
        gt4 = gt[meanedge > self.divide[2]]
        gt5 = gt[meanedge > self.divide[3]]
        return [gt1, gt2, gt3, gt4, gt5]
        
    def encode(self, gt):
        # gt为 numpy形式, shape = (N, 5) loc = x1,y1,x2,y2,cls  cls为(0-19)
        gt = self.gt_process(gt) #gt为list,存储5个layer的gt
        Reg = np.zeros(shape = (0,4))
        Center = np.zeros(shape = (0,))
        Cls = np.zeros(shape = (0,self.clsNum))
        for i in range(self.layerNum):
            #if(gt[i].shape[0] == 0):continue
            targetReg = np.zeros(shape = self.size2layer[i] + [4], dtype = np.float32)#4分别为（l, r, t, b）
            targetCenter = np.zeros(shape = self.size2layer[i], dtype = np.float32)
            targetCls = np.zeros(shape = self.size2layer[i] + [self.clsNum], dtype = np.float32)
            gt_cls = gt[i][:,4].astype(np.int32)
            #print("layer = %d, gt[i] = %s"%(i+1, gt[i]))
            targetGt = (gt[i][:,:4]/self.stride[i])
            targetGt[:,[2,3]] = np.ceil(targetGt[:,[2,3]])
            targetGt[:,[0,1]] = np.floor(targetGt[:,[0,1]])
            targetGt = targetGt.astype(np.int32)
            #[w, h] = targetGt[:,2] - targetGt[:,0] + 1, targetGt[:,3] - targetGt[:,1] + 1
            w, h = targetGt[:,2] - targetGt[:,0], targetGt[:,3] - targetGt[:,1]
            for j in range(targetGt.shape[0]):
                bbox = targetGt[j]
                bboxMap = np.zeros(shape = (h[j], w[j], 4))
                bboxMap[:,:,0] = np.arange(w[j]).reshape(1, -1) * np.ones(shape = (h[j], 1))
                bboxMap[:,:,1] = w[j] - 1 - bboxMap[:,:,0]
                bboxMap[:,:,2] = np.arange(h[j]).reshape(-1, 1) * np.ones(shape = (1, w[j]))
                bboxMap[:,:,3] = h[j] - 1 - bboxMap[:,:,2]
                bboxMap += 1
                #try:
                targetReg[bbox[1]:bbox[3], bbox[0]:bbox[2]] = bboxMap
                #except:
                #    print("bbox ",bbox)
                #    print("targetReg ",targetReg.shape)
                #    print("bboxMap ",bboxMap.shape)
                targetCenter[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.sqrt(
                    (np.minimum(bboxMap[:,:,0], bboxMap[:,:,1]) * np.minimum(bboxMap[:,:,2], bboxMap[:,:,3])) / (np.maximum(bboxMap[:,:,0], bboxMap[:,:,1]) * np.maximum(bboxMap[:,:,2], bboxMap[:,:,3]))
                )
                targetCls[bbox[1]:bbox[3], bbox[0]:bbox[2], gt_cls[j]] = 1
            Reg = np.vstack((Reg, targetReg.reshape(-1, 4)))
            Cls = np.vstack((Cls, targetCls.reshape(-1, self.clsNum)))
            Center = np.hstack((Center, targetCenter.reshape(-1)))
        return Cls, Reg, Center


# In[12]:


category_id = {"破洞":1,"水渍":2,"油渍":2,"污渍":2,
               "三丝":3,"结头":4,"花板跳":5,"百脚":6,
               "毛粒":7,"粗经":8,"松经":9,"断经":10,
               "吊经":11,"粗维":12,"纬缩":13,"浆斑":14,
               "整经结":15,"星跳":16,"跳花":16,"断氨纶":17,
               "稀密档":18,"浪纹档":18,"色差档":18,"磨痕":19,
               "轧痕":19,"修痕":19,"烧毛痕":19,"死皱":20,
               "云织":20,"双纬":20,"双经":20,"跳纱":20,
               "筘路":20,"纬纱不良":20}
category_id = {key:category_id[key]-1 for key in category_id}  
img_h = 1000 - 1
img_w = 2446 - 1
clsType = [i for i in category_id]
with open(r"work/train_data/guangdong1_round1_train1_20190818/Annotations/anno_train.json",'r') as f:
    js = json.load(f)

defect_path = r"work/train_data/guangdong1_round1_train1_20190818/defect_Images"
normal_path = r"work/train_data/guangdong1_round1_train1_20190818/normal_Images"

id2category = {0:"破洞", 1:random.choice(["水渍","油渍","污渍"]),
               2:"三丝", 3:"结头", 4:"花板跳",
               5:"百脚", 6:"毛粒", 7:"粗经",
               8:"松经", 9:"断经",
               10:"吊经", 11:"粗维", 12:"纬缩", 13:"浆斑",
               14:"整经结",15:random.choice(["星跳","跳花"]),
               16:"断氨纶",17:random.choice(["稀密档","浪纹档","色差档"]),
               18:random.choice(["磨痕","轧痕","修痕","烧毛痕"]),
               19:random.choice(["死皱","云织","双纬","双经","跳纱","筘路","纬纱不良"])}
               
def bufferGenerater(js, defect_path, size = 3):
    buffer = {i:[] for i in category_id}    #buffer中为【x1,y1,x2,y2,w,h】
    total = len(buffer) * size
    count = 0
    for info in js:
        defect_name = info["defect_name"]
        if(len(buffer[defect_name]) < size):
            count += 1
            img = cv2.imread(os.path.join(defect_path, info["name"]))
            bbox = [int(i) for i in info["bbox"]]
            #if(bbox[2]>2445):bbox[2] = 2445
            #if(bbox[3]> 999):bbox[3] = 999
            dat = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            buffer[defect_name].append(dat)
        if count >= total:break
    return buffer
Buffer = bufferGenerater(js, defect_path, size = 5)

def js2label(js, normlist):
    label = {}
    for info in js:
        if not info["name"] in label:
            label[info["name"]] = []
        label[info["name"]].append(info["bbox"] + [category_id[info["defect_name"]]])
        
    for key in label:
        temp = np.array(label[key]) #将bbox转为numpy类型,并去小数
        temp[:,[2,3]] = np.ceil(temp[:,[2,3]])
        temp[:,2] = np.clip(temp[:,2], 0, 2445)
        temp[:,3] = np.clip(temp[:,3], 0, 999)
        temp[:,[0,1]] = np.floor(temp[:,[0,1]])
        label[key] = temp.astype(np.int32) 

    for imgname in normlist:
        label[imgname] = None
    return label

encode = Encode() ##实例化Encode

mirror = 0.35
flip = 0.35
mixup = 0.35
def aug(img, anno):
    #draw(img.copy(), anno.copy())
    #print("after aug ",anno)
    if(random.random() < mirror):
        img = cv2.flip(img, 1)
        if(anno.shape[0] > 0):
            w = anno[:,2] - anno[:,0]
            anno[:,2] = img_w - anno[:,0] + 1
            anno[:,0] = anno[:,2] - w
        
    if(random.random() < flip):
        img = cv2.flip(img, 0)
        if(anno.shape[0] > 0):
            h = anno[:,3] - anno[:,1]
            anno[:,3] = img_h - anno[:,1] + 1
            anno[:,1] = anno[:,3] - h
    #draw(img.copy(), anno.copy())
    #print("befor aug ",anno)
    return img,anno

def normalImgAddSample(img):
    global Buffer
    temp = np.zeros_like(img, dtype = np.int32)
    highlight = (img>220)
    img_mean = np.mean(img, axis = (0,1))
    annos = []
    for i in range(5):
        add_type = random.choice(clsType)
        dif = []
        for buf_img in Buffer[add_type]:
            mean = np.mean(np.array(buf_img), axis = (0,1))
            dif.append(np.sum((img_mean - mean)**2))
        index = np.argmin(dif)
        add_img = Buffer[add_type][index]
        [h,w] = add_img.shape[:2]
        
        if(h >= img_h):index_h = 0
        else:index_h = random.randint(0,img_h - h)

        if(w >= img_w):index_w = 0     
        else:index_w = random.randint(0,img_w - w)

        if(np.sum(temp[index_h:index_h + h, index_w:index_w + w]) == 0 and            np.sum(highlight[index_h:index_h + h, index_w:index_w + w]) <= 0.1 * h * w):
            cover_area_mean = np.mean(img[index_h:index_h + h, index_w:index_w + w], axis = (0,1))
            #if(np.mean(cover_area_mean) > 170):continue
            temp[index_h:index_h + h, index_w:index_w + w] = add_img * (cover_area_mean / np.mean(add_img, axis = (0,1)))
            bbox = [index_w, index_h, w, h, category_id[add_type]]
            annos.append(bbox)
    
    if(len(annos) == 0):return img, np.zeros((0,5))
    annos = np.array(annos)
    annos[:,2] += annos[:,0]
    annos[:,3] += annos[:,1]
    img = img * (temp == 0) + temp 
    return img,annos
 

def generator(defect_path = None, normal_path = None, label = None, imgnames = None, normlist = None):
    def __generater__():
        for name in imgnames:
            if False:
            #if label[name] is None:
                x = cv2.imread(os.path.join(normal_path, name))
                x,y = normalImgAddSample(x)
            else:
                x = cv2.imread(os.path.join(defect_path, name))
                y = label[name]
                
                """
                if(random.random() < 0.05):
                    index = y.shape[0]%3 - 1
                    cls = id2category[y[index][4]]
                    Buffer[cls].pop(0)
                    bbox = y[index]
                    Buffer[cls].append(x[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                """
                if(random.random() < mixup):
                    tempImg = cv2.imread(os.path.join(normal_path, random.choice(normlist)))
                    x = (x*0.95 + tempImg*0.05).astype(np.uint8)
            x,y = aug(x, y)
            Cls, Reg, Center = encode.encode(y)
            yield x.transpose(2,0,1), Cls, Reg, Center
            
    return __generater__
   
label = js2label(js, os.listdir(r"work/train_data/guangdong1_round1_train1_20190818/normal_Images"))


# In[13]:


#import copy
#train_reader = generator(defect_path=defect_path, normal_path=normal_path, label=label)
#for idx, dat in enumerate(train_reader()):
#    data1 = copy.deepcopy(dat)
#    print(idx)


# In[1]:


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
from data import datGenerator

batch_size = 1
epochs = 15
learning_rate = 3e-5
start = time.time()
cls_loss_hist = collections.deque(maxlen=300)
reg_loss_hist = collections.deque(maxlen=300)
cen_loss_hist = collections.deque(maxlen=300)

#train_reader = generator(defect_path=defect_path, normal_path=normal_path, label=label, imgnames = imgnames, normlist = normlist)
#train_reader = paddle.reader.shuffle(train_reader, buf_size=32)
#train_reader = paddle.batch(train_reader, batch_size= batch_size,drop_last=False)
train_generator = datGenerator()
train_reader = train_generator.generator
train_reader = paddle.batch(train_reader, batch_size= batch_size,drop_last=False)

with fluid.dygraph.guard():
    lr = fluid.layers.cosine_decay( learning_rate = learning_rate, step_each_epoch=4774, epochs=epochs)
    adam = fluid.optimizer.AdamOptimizer(learning_rate = lr)
    model = FCOS("fcos", 20, batch=batch_size)
    parameters, adam_parm = fluid.dygraph.load_persistables("work/Model/model1/epochs=2")
    adam.load(adam_parm)
    model.load_dict(parameters)
    print("training begin.....")
    for epoch in range(3,epochs):
        for idx, data in enumerate(train_reader()):
            img = fluid.dygraph.to_variable(np.stack([dat[0] for dat in data], axis=0).astype(np.float32))
            cls = fluid.dygraph.to_variable(np.stack([dat[1] for dat in data], axis=0).astype(np.float32))
            reg = np.stack([dat[2] for dat in data], axis=0)
            reg_mask = (reg[:,:,0]> 0)
            reg = fluid.dygraph.to_variable(reg.astype(np.float32))
            reg_mask = fluid.dygraph.to_variable(reg_mask.astype(np.float32))
            cen = fluid.dygraph.to_variable(np.stack([dat[3] for dat in data], axis=0).astype(np.float32))
            cls.stop_gradient = True
            reg.stop_gradient = True
            cen.stop_gradient = True
            reg_mask.stop_gradient = True
    
            focal_loss, iou_loss, centerness_loss = model(img, cls, reg, cen, reg_mask)
            loss = focal_loss + iou_loss + centerness_loss
            loss.backward()
            adam.minimize(loss)
            cls_loss_hist.append(focal_loss.numpy())
            reg_loss_hist.append(iou_loss.numpy())
            cen_loss_hist.append(centerness_loss.numpy())
            if(idx % 100 == 0):
                cls_loss_mean = np.mean(cls_loss_hist)
                reg_loss_mean = np.mean(reg_loss_hist)
                cen_loss_mean = np.mean(cen_loss_hist)
                loss_mean = cls_loss_mean + reg_loss_mean + cen_loss_mean
                print("epoch = %d | iter = %d | loss = %.5f | focal_loss = %.5f | iou_loss = %.5f | centerness_loss = %.5f | use time = %.3f s"%(epoch, idx, loss_mean, cls_loss_mean, reg_loss_mean, cen_loss_mean, time.time() - start))
                start = time.time()
            model.clear_gradients()
        fluid.dygraph.save_persistables(model.state_dict(), "work/Model/model1/epochs=%s"%epoch, optimizers = adam)


# In[5]:


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
print(time.asctime( time.localtime(time.time()) )[11:])
batch_size = 1
def nms(data, threshold):
    #data type = numpy ,shape = N,6   [x1, y1, x2, y2, score, cls]
    #print("before ",data)
    scores = data[:,4]
    x1 = data[:,0]
    y1 = data[:,1]
    x2 = data[:,2]
    y2 = data[:,3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    index = scores.argsort()[::-1]
    res = []
    while index.shape[0] >0:
        bbox = data[index[0]]
        res.append(bbox)
        x11 = np.maximum(bbox[0], x1[index[1:]])
        y11 = np.maximum(bbox[1], y1[index[1:]])
        x22 = np.minimum(bbox[2], x2[index[1:]])
        y22 = np.maximum(bbox[3], y2[index[1:]])
        
        w = np.maximum(0, x22 - x11 + 1) 
        h = np.maximum(0, y22 - y11 + 1)
        i = w*h
        iou = i / (areas[index[0]] + areas[index[1:]] - i)
        idx = iou < threshold
        index = index[1:][idx]
        
    #print("after ",np.array(res))   
    return res

def testGenerator(testPath):
    testList = os.listdir(testPath)
    #with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train.json", 'r') as f:
    #        js = json.load(f)
    #testList = []
    #testPath = r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"
    #for info in js:
    #    if info["name"] not in testList:
    #        testList.append(info["name"])
    def __testImg__():
        for imgname in testList:
            img = cv2.imread(os.path.join(testPath, imgname))
            yield img.transpose(2,0,1), imgname
    return  __testImg__
    
test_reader = testGenerator(r"work/test_data/guangdong1_round1_testA_20190818")
test_reader = paddle.batch(test_reader, batch_size= 2)

start = time.time()
with fluid.dygraph.guard():
    model = FCOS("fcos", 20, batch=batch_size, is_trainning=False)
    parameters, _ = fluid.dygraph.load_persistables("work/model/v4/7head_eps = 10_half")
    model.load_dict(parameters)
    model.eval()
    res = {}
    for testData in test_reader():
        img = fluid.dygraph.to_variable(np.stack([dat[0] for dat in testData], axis=0).astype(np.float32))
        imgname = [dat[1] for dat in testData]
        Cls, Scores, Centerness, Loc = model(img)
        for i in imgname:
            res[i] = np.zeros(shape = (0,6))  #[x1,y1,x2,y2, score, cls]
        feat_size = [(125, 306), (63, 153), (32, 77), (16, 39), (8, 20)]
        stride = [8, 16, 32, 64, 128]
        threshold = 0.45

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
                bbox = np.dstack([bbox * stride[i], score[batch_idx], b_cls])
                bbox = bbox[b_idx]
                bbox = np.clip(bbox,0,2446)
                bbox[:,3] = np.clip(bbox[:,3],0,1000)
                res[imgname[batch_idx]] = np.vstack((res[imgname[batch_idx]], bbox))
                
    ans = []
    for key in res:
        if(res[key].shape[0]>0):
            bbox = nms(res[key], 0.2)
            bbox = np.round(bbox,2)
            for i in range(len(bbox)):
                temp = {}
                temp["name"] = key
                temp["category"] = int(bbox[i][5]) + 1
                temp["bbox"] = list(bbox[i][:4])
                temp["score"] = bbox[i][4]
                ans.append(temp)
                
    with open('work/res/eps10_half.json', 'w') as fp:
        json.dump(ans, fp, indent=4, separators=(',', ': '))
print(time.time() - start)


# In[1]:


import cv2
import numpy as np
import json
import os
import random
from matplotlib import pyplot as plt 
import copy
import collections
with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train.json",'r') as f:
    js_old = json.load(f)
    
with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train2.json",'r') as f:
    js_new = json.load(f)
    
defect_path = r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"
normal_path = r"work/train_dat/guangdong1_round1_train1_20190818/normal_Images"
js = js_old + js_new

category2id = {"破洞":1,"水渍":2,"油渍":2,"污渍":2,
               "三丝":3,"结头":4,"花板跳":5,"百脚":6,
               "毛粒":7,"粗经":8,"松经":9,"断经":10,
               "吊经":11,"粗维":12,"纬缩":13,"浆斑":14,
               "整经结":15,"星跳":16,"跳花":16,"断氨纶":17,
               "稀密档":18,"浪纹档":18,"色差档":18,"磨痕":19,
               "轧痕":19,"修痕":19,"烧毛痕":19,"死皱":20,
               "云织":20,"双纬":20,"双经":20,"跳纱":20,
               "筘路":20,"纬纱不良":20}

id2category = {1:"破洞", 2:random.choice(["水渍","油渍","污渍"]),
               3:"三丝", 4:"结头", 5:"花板跳",
               6:"百脚", 7:"毛粒", 8:"粗经",
               9:"松经", 10:"断经",
               11:"吊经", 12:"粗维", 13:"纬缩", 14:"浆斑",
               15:"整经结",16:random.choice(["星跳","跳花"]),
               17:"断氨纶",18:random.choice(["稀密档","浪纹档","色差档"]),
               19:random.choice(["磨痕","轧痕","修痕","烧毛痕"]),
               20:random.choice(["死皱","云织","双纬","双经","跳纱","筘路","纬纱不良"])}
img_label = {}
for info in js:
    if info["name"] not in img_label:
        img_label[info["name"]] = []
    img_label[info["name"]].append(info["bbox"] + [category2id[info["defect_name"]]])
    
cls_num = [0]*20
cls_type = [""] * 20  #[i for i in range(20)]
for cls in category2id:
    index = category2id[cls]
    cls_type[index - 1] += cls + " "
cls_type[19] = cls_type[19][:10]    
for info in js:
    idx = category2id[info["defect_name"]] - 1
    cls_num[idx] += 1

mean_area = [[] for i in range(20)]
for info in js:
    cls = info["defect_name"]
    bbox = info["bbox"]
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    mean_area[category2id[cls] - 1].append(area)
mean_area = [np.mean(area) for area in mean_area]
mean_area = np.sqrt(mean_area)

pix_num = ((mean_area < 80) * (mean_area / 8) * cls_num) + ((mean_area < 160) * (mean_area > 80)* (mean_area / 16) * cls_num) + ((mean_area < 320) * (mean_area > 160) * (mean_area / 32) * cls_num) + ((mean_area < 640) * (mean_area > 320)  * (mean_area / 64) * cls_num) + ((mean_area > 640) * (mean_area / 128) * cls_num)
add_num = (np.max(pix_num)/pix_num) * ((np.array(mean_area)<50) + 1)
add_num = np.round(add_num * 0.8).astype(np.int32)

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
    cls = bbox[:,-1]
    add_times = []
    for i in range(cls.shape[0]):
        add_times.append(add_num[cls[i] - 1])
    add_times = int(np.mean(add_times))
    if(add_times == 1):continue
    img = cv2.imread(os.path.join(src_path, imgname))
    count.append(add_times)
    for i in range(1, add_times):
        new_img_name = imgname[:-4] + "_" + str(i) + ".jpg"
        new_img = copy.deepcopy(img)
        new_bbox = copy.deepcopy(bbox)
        cut_h = random.randint(0,100)
        cut_w = random.randint(0,200)
        new_img = new_img[cut_h:, cut_w:]
        new_bbox[:,[0,2]] = new_bbox[:,[0,2]] - cut_w
        new_bbox[:,[1,3]] = new_bbox[:,[1,3]] - cut_h
        if(np.sum(new_bbox < 0) > 0):continue
        scale = np.array([1000,2446]) / new_img.shape[:2]
        new_img = cv2.resize(new_img, (2446, 1000))
        new_bbox[:,[0,2]] = new_bbox[:,[0,2]] * scale[1]
        new_bbox[:,[1,3]] = new_bbox[:,[1,3]] * scale[0]
        while(True):
            norm_add_img = cv2.imread(os.path.join(norm_path, random.choice(norm_list)))
            if(np.sum(norm_add_img > 200) < 104600):
                break
        new_img_mean = np.mean(new_img, axis = (0,1))
        add_img_mean = np.mean(norm_add_img, axis = (0,1))
        new_img = (new_img*0.85 + norm_add_img*0.15 * (new_img_mean/add_img_mean) ).astype(np.uint8)
        add_img_label[new_img_name] = new_bbox
        cv2.imwrite(os.path.join(target_path, new_img_name), new_img)
    if(idx  % 200 == 0):
        print("idx = %d, mean_times = %d"%(idx, np.mean(count)))


# In[3]:


print("final")
add_js = []
for imgname in add_img_label:
    anno = add_img_label[imgname].astype(np.float)
    for i in range(anno.shape[0]):
        temp = {}
        temp["bbox"] = list(anno[i])[:4]
        temp["name"] = imgname
        temp["defect_name"] = id2category[anno[i][-1]]
        add_js.append(temp)
with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train_add.json",'w') as fp:
    json.dump(add_js, fp, indent=4, separators=(',', ': '))


# In[7]:


with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train_add.json") as fp:
    js1 = json.load(fp)
with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train.json") as fp:
    js2 = json.load(fp)
with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/anno_train2.json") as fp:
    js3 = json.load(fp)
js = js1 + js2 + js3


# In[12]:


with open(r"work/train_dat/guangdong1_round1_train1_20190818/Annotations/label.json",'w') as fp:
    json.dump(js, fp, indent=4, separators=(',', ': '))


# In[6]:


len(os.listdir(r"work/train_dat/guangdong1_round1_train1_20190818/defect_Images"))


# In[ ]:




