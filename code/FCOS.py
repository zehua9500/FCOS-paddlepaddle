import paddle.fluid as fluid
import paddle
import numpy as np
from ResNet import ResNet
from loss import Loss


class ASPP(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test):
        super(ASPP, self).__init__(name_scope)
        self.dilate1 = fluid.dygraph.Conv2D(name_scope + "_dilate1", num_filters=128, filter_size=3, stride=1, padding=1, dilation=1)
        self.dilate2 = fluid.dygraph.Conv2D(name_scope + "_dilate2", num_filters=128, filter_size=3, stride=1, padding=2, dilation=2)
        # self.dilate3 = fluid.dygraph.Conv2D(name_scope + "_dilate3", num_filters=64, filter_size=3, stride=1, padding=4, dilation=4)
        # self.dilate4 = fluid.dygraph.Conv2D(name_scope + "_dilate4", num_filters=64, filter_size=3, stride=1, padding=6, dilation=6)
        self.bn1 = fluid.dygraph.BatchNorm(name_scope + "_bn1", 256, act="relu", is_test=is_test)

        self.merge = fluid.dygraph.Conv2D(name_scope + "_merge", num_filters=256, filter_size=1, stride=1)
        self.bn2 = fluid.dygraph.BatchNorm(name_scope + "_bn2", 256, act="relu", is_test=is_test)

    def forward(self, inputs):
        x1 = self.dilate1(inputs)
        x2 = self.dilate2(inputs)

        # x3 = self.dilate3(inputs)
        # x4 = self.dilate4(inputs)
        out = fluid.layers.concat(input=[x1, x2], axis=1)
        out = self.bn1(out)
        out = self.merge(out)
        return self.bn2(out)


class Head(fluid.dygraph.Layer):
    def __init__(self, name_scope, clsNum, is_test):
        super(Head, self).__init__(name_scope)
        self.cls1 = ASPP(name_scope + "_cls1", is_test=is_test)
        self.cls2 = ASPP(name_scope + "_cls2", is_test=is_test)
        self.cls3 = ASPP(name_scope + "_cls3", is_test=is_test)
        self.cls4 = ASPP(name_scope + "_cls4", is_test=is_test)
        self.cls5 = fluid.dygraph.Conv2D(name_scope + "_cls5", num_filters=256, filter_size=3, stride=1, padding=1)
        self.cls6 = fluid.dygraph.Conv2D(name_scope + "_cls6", num_filters=256, filter_size=3, stride=1, padding=1)
        self.cls7 = fluid.dygraph.Conv2D(name_scope + "_cls7", num_filters=256, filter_size=3, stride=1, padding=1)
        self.cls8 = fluid.dygraph.Conv2D(name_scope + "_cls8", num_filters=256, filter_size=3, stride=1, padding=1)
        # self.cen1 = ASPP("cen1", is_test=is_test)
        # self.cen2 = ASPP("cen2", is_test=is_test)
        # self.cen3 = ASPP("cen3", is_test=is_test)
        # self.cen4 = ASPP("cen4", is_test = is_test)

        self.loc1 = ASPP(name_scope + "_loc1", is_test=is_test)
        self.loc2 = ASPP(name_scope + "_loc2", is_test=is_test)
        self.loc3 = ASPP(name_scope + "_loc3", is_test=is_test)
        self.loc4 = ASPP(name_scope + "_loc4", is_test=is_test)
        self.loc5 = fluid.dygraph.Conv2D(name_scope + "_loc5", num_filters=256, filter_size=3, stride=1, padding=1)
        self.loc6 = fluid.dygraph.Conv2D(name_scope + "_loc6", num_filters=256, filter_size=3, stride=1, padding=1)
        self.loc7 = fluid.dygraph.Conv2D(name_scope + "_loc7", num_filters=256, filter_size=3, stride=1, padding=1)
        self.loc8 = fluid.dygraph.Conv2D(name_scope + "_loc8", num_filters=256, filter_size=3, stride=1, padding=1)

        # self.cls_out_head = ASPP("cls_out_head", is_test = is_test)
        # self.center_ness_head = ASPP("center_ness_head", is_test = is_test)
        # self.regression_head = ASPP("regression_head", is_test = is_test)
        self.cls_out = fluid.dygraph.Conv2D(name_scope + "_class", num_filters=clsNum, filter_size=3, stride=1, padding=1, dilation=1)
        self.center_ness = fluid.dygraph.Conv2D(name_scope + "_center_ness", num_filters=1, filter_size=3, stride=1, padding=1,
                                                dilation=1)
        self.regression = fluid.dygraph.Conv2D(name_scope + "_regression", num_filters=4, filter_size=3, stride=1, padding=1,
                                               dilation=1)

    def forward(self, inputs):
        cls = self.cls1(inputs)
        cls = self.cls2(cls)
        cls = self.cls3(cls)
        cls = self.cls4(cls)

        cls_residual = self.cls5(cls)
        cls_residual = self.cls6(cls_residual)
        cls_residual = self.cls7(cls_residual)
        cls_residual = self.cls8(cls_residual)
        cls = cls + cls_residual
        # cls_out = self.cls_out_head(cls)

        cls_out = self.cls_out(cls)

        # cen = self.cen1(inputs)
        # cen = self.cen2(cen)
        # cen = self.cen3(cen)
        # cen = self.cen4(cen)
        # center_ness = self.center_ness_head(cen)
        center_ness = self.center_ness(cls)

        loc = self.loc1(inputs)
        loc = self.loc2(loc)
        loc = self.loc3(loc)
        loc = self.loc4(loc)

        loc_resdual = self.loc5(loc)
        loc_resdual = self.loc6(loc_resdual)
        loc_resdual = self.loc7(loc_resdual)
        loc_resdual = self.loc8(loc_resdual)
        loc = loc + loc_resdual

        loc = self.regression(loc)

        return [fluid.layers.sigmoid(cls_out),
                fluid.layers.sigmoid(center_ness),
                fluid.layers.exp(loc)]


class FCOS(fluid.dygraph.Layer):
    def __init__(self, name_scope, clsNum, batch=8, is_trainning=True):
        super(FCOS, self).__init__(name_scope)

        self.trainning = is_trainning
        self.resnet = ResNet(name_scope + "_ResNet", is_test=not is_trainning)
        self.head = Head(name_scope + '_Head1', clsNum, is_test=not is_trainning)  # head1用于前3层 fpn
        # self.head2 = Head(name_scope + '_Head2', clsNum, is_test=not is_trainning)  # head2用于后2层 fpn
        self.stride = [8, 16, 32, 64, 128]
        self.clsNum = clsNum
        self.batch = batch
        self.loss = Loss(name_scope + "_loss")
        self.size2layer = [[125, 306], [63, 153], [32, 77], [16, 39], [8, 20]]  # 各fpn层的featuremap size  因为在布匹检测中输入尺寸都相同。故用相同feature map
        # self.area = [0, 38250, 47889, 50353, 50977, 51137]
        self.feat_map = []
        for i in range(5):
            temp = np.zeros(shape=(self.size2layer[i] + [2]))
            temp[:, :, 0] = np.arange(self.size2layer[i][1]).reshape(1, -1) * self.stride[i] + self.stride[i] / 2
            temp[:, :, 1] = np.arange(self.size2layer[i][0]).reshape(-1, 1) * self.stride[i] + self.stride[i] / 2
            self.feat_map.append(temp)
        print("FCOS load final")

    def forward(self, x, cls_label=None, reg_label=None, cent_label=None, cen_mask=None, mask=None, gt=None):
        feature = self.resnet(x)
        # Cls = paddle.fluid.layers.zeros(shape = (self.batch, 0, self.clsNum), dtype = "float32")
        # Reg = paddle.fluid.layers.zeros(shape = (self.batch, 0, 4), dtype = "float32")
        # Center = paddle.fluid.layers.zeros(shape = (self.batch, 0), dtype = "float32")
        Cls, Reg, Center = [], [], []
        two_stage_input = []
        if (self.trainning):
            for idx, feat in enumerate(feature):
                # if (idx < 3):
                cls_out, center_ness, loc = self.head(feat)

                # else:
                #    cls_out, center_ness, loc = self.head2(feat)
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
            # return self.loss(Cls, cls_label, Reg, reg_label, Center, cent_label, reg_mask)
            return self.loss(Cls, cls_label, Reg, reg_label, Center, cent_label, cen_mask, mask)
        else:
            Scores = []
            for idx, feat in enumerate(feature):
                # if (idx < 3):
                cls_out, center_ness, loc = self.head(feat)
                # else:
                #    cls_out, center_ness, loc = self.head2(feat)
                cls_out = fluid.layers.transpose(cls_out, perm=[0, 2, 3, 1])
                center_ness = fluid.layers.transpose(center_ness, perm=[0, 2, 3, 1])
                loc = fluid.layers.transpose(loc, perm=[0, 2, 3, 1])

                argmax = fluid.layers.argmax(cls_out, axis=3)
                score = fluid.layers.reduce_max(cls_out, dim=3, keep_dim=True)
                Scores.append(score)
                Cls.append(argmax)
                Reg.append(loc)
                Center.append(center_ness)
            return Cls, Scores, Center, Reg
