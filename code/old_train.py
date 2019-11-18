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
epochs = 10
learning_rate = 5e-5
start = time.time()

cls_loss_hist = collections.deque(maxlen=300)
reg_loss_hist = collections.deque(maxlen=300)
cen_loss_hist = collections.deque(maxlen=300)

lr = fluid.layers.cosine_decay(learning_rate=learning_rate, step_each_epoch=6436 + 400, epochs=21)
adam = fluid.optimizer.AdamOptimizer(learning_rate=lr)

train_generator = datGenerator()
train_reader = train_generator.generator
train_reader = paddle.batch(train_reader, batch_size=batch_size, drop_last=False)

with fluid.dygraph.guard():
    model = FCOS("fcos", 20, batch=batch_size)
    parameters, _ = fluid.dygraph.load_persistables("work/model/v1/epochs_warmUp")
    model.load_dict(parameters)
    print("training begin.....")
    for epoch in range(1, epochs + 1):
        for idx, data in enumerate(train_reader()):
            img = fluid.dygraph.to_variable(np.stack([dat[0] for dat in data], axis=0))
            cls = fluid.dygraph.to_variable(np.stack([dat[1] for dat in data], axis=0))
            reg = np.stack([dat[2] for dat in data], axis=0)
            # reg_mask = (reg[:,:,0]> 0)
            reg = fluid.dygraph.to_variable(reg)
            # reg_mask = fluid.dygraph.to_variable(reg_mask.astype(np.float32))
            cen = fluid.dygraph.to_variable(np.stack([dat[3] for dat in data], axis=0).astype(np.float32))
            cls.stop_gradient = True
            reg.stop_gradient = True
            cen.stop_gradient = True
            # reg_mask.stop_gradient = True

            # f_loss, i_loss, c_loss = model(img, cls, reg, cen, reg_mask)
            f_loss, i_loss, c_loss = model(img, cls, reg, cen)
            loss = f_loss + i_loss + c_loss
            loss.backward()
            adam.minimize(loss)
            cls_loss_hist.append(f_loss.numpy())
            reg_loss_hist.append(i_loss.numpy())
            cen_loss_hist.append(c_loss.numpy())
            if (idx % 100 == 0):
                cls_loss_mean = np.mean(cls_loss_hist)
                reg_loss_mean = np.mean(reg_loss_hist)
                cen_loss_mean = np.mean(cen_loss_hist)
                loss_mean = cls_loss_mean + reg_loss_mean + cen_loss_mean
                # print(time.asctime( time.localtime(time.time()) )[11:])
                print("epoch = %d | iter = %d | loss = %.5f | focal_loss = %.5f | iou_loss = %.5f | centerness_loss = %.5f | use time = %.3f s | time is %s" % \
                      (epoch, idx, loss_mean, cls_loss_mean, reg_loss_mean, cen_loss_mean, time.time() - start, time.asctime(time.localtime(time.time()))[11:]))
                start = time.time()
            model.clear_gradients()
        fluid.dygraph.save_persistables(model.state_dict(), "work/model/v1/epochs=%s" % epoch)
