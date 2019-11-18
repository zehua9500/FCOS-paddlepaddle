import paddle.fluid as fluid
import paddle


class SEConcat(fluid.dygraph.Layer):
    def __init__(self, name_scope, channel=256, is_test=False):
        super(SEConcat, self).__init__(name_scope)
        self.downChannel = fluid.dygraph.Conv2D(name_scope + "_downChannel", 256, filter_size=1, stride=1)
        self.fc = fluid.dygraph.FC(name_scope + "_fc", size=256, act="sigmoid", is_test=is_test, dtype='float32')

    def forward(self, x1, x2):
        x = fluid.layers.concat(input=[x1, x2], axis=1)
        x = self.downChannel(x)
        mean = fluid.layers.reduce_mean(x, dim=[2, 3])
        mean = self.fc(mean)
        return fluid.layers.elementwise_mul(x, mean, axis=0)


class FPN(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test, channel=256):
        super(FPN, self).__init__(name_scope)
        self.P8_1 = fluid.dygraph.Conv2D(name_scope + "_P8_1", channel * 2, filter_size=3, stride=2, padding=1)
        self.bn8_1 = fluid.dygraph.BatchNorm(name_scope + "_bn8_1", channel * 2, act="relu", is_test=is_test)
        self.P8_2 = fluid.dygraph.Conv2D(name_scope + "_P8_2", channel, filter_size=3, stride=2, padding=1)
        # self.bn8_2 = fluid.dygraph.BatchNorm(name_scope + "_bn8_2", channel, act="relu", is_test=is_test)

        self.P7_1 = fluid.dygraph.Conv2D(name_scope + "_P7_1", channel, filter_size=3, stride=1, padding=1)
        self.P7_2 = fluid.dygraph.Conv2D(name_scope + "_P7_2", channel, filter_size=3, stride=1, padding=1)
        self.P7_up = fluid.dygraph.Conv2D(name_scope + "_P7_up", channel, filter_size=1, stride=1)
        self.bn7_1 = fluid.dygraph.BatchNorm(name_scope + "_bn7_1", channel, act="relu", is_test=is_test)
        self.bn7_2 = fluid.dygraph.BatchNorm(name_scope + "_bn7_2", channel, act="relu", is_test=is_test)

        self.P6_1 = fluid.dygraph.Conv2D(name_scope + "_P6_1", channel, filter_size=3, stride=1, padding=1)
        self.P6_2 = fluid.dygraph.Conv2D(name_scope + "_P6_2", channel, filter_size=3, stride=1, padding=1)
        self.P6_up = fluid.dygraph.Conv2D(name_scope + "_P6_up", channel, filter_size=1, stride=1)
        self.bn6_1 = fluid.dygraph.BatchNorm(name_scope + "_bn6_1", channel, act="relu", is_test=is_test)
        self.bn6_2 = fluid.dygraph.BatchNorm(name_scope + "_bn6_2", channel, act="relu", is_test=is_test)

        self.P5_1 = fluid.dygraph.Conv2D(name_scope + "_P5_1", channel, filter_size=3, stride=1, padding=1)
        self.P5_2 = fluid.dygraph.Conv2D(name_scope + "_P5_2", channel, filter_size=3, stride=1, padding=1)
        self.P5_up = fluid.dygraph.Conv2D(name_scope + "_P5_up", channel, filter_size=1, stride=1)
        self.bn5_1 = fluid.dygraph.BatchNorm(name_scope + "_bn5_1", channel, act="relu", is_test=is_test)
        self.bn5_2 = fluid.dygraph.BatchNorm(name_scope + "_bn5_2", channel, act="relu", is_test=is_test)

        self.P4_1 = fluid.dygraph.Conv2D(name_scope + "_P4_1", channel, filter_size=3, stride=1, padding=1)
        self.P4_2 = fluid.dygraph.Conv2D(name_scope + "_P4_2", channel, filter_size=3, stride=1, padding=1)
        self.P4_up = fluid.dygraph.Conv2D(name_scope + "_P4_up", channel, filter_size=1, stride=1)
        self.bn4_1 = fluid.dygraph.BatchNorm(name_scope + "_bn4_1", channel, act="relu", is_test=is_test)
        self.bn4_2 = fluid.dygraph.BatchNorm(name_scope + "_bn4_2", channel, act="relu", is_test=is_test)

        self.P3_1 = fluid.dygraph.Conv2D(name_scope + "_P3_1", channel, filter_size=3, stride=1, padding=1)
        self.P3_2 = fluid.dygraph.Conv2D(name_scope + "_P3_2", channel, filter_size=3, stride=1, padding=1)
        self.P3_up = fluid.dygraph.Conv2D(name_scope + "_P3_up", channel, filter_size=1, stride=1)
        self.bn3_1 = fluid.dygraph.BatchNorm(name_scope + "_bn3_1", channel, act="relu", is_test=is_test)
        self.bn3_2 = fluid.dygraph.BatchNorm(name_scope + "_bn3_2", channel, act="relu", is_test=is_test)

    def forward(self, inputs):
        C3, C4, C5, C6, C7 = inputs

        # P8_x = self.bn8_1(C7)
        P8_x = self.P8_1(C7)
        P8_x = self.bn8_1(P8_x)
        P8_x = self.P8_2(P8_x)
        P8_upsample = fluid.layers.resize_nearest(input=P8_x, scale=None, out_shape=(8, 20))

        P7_x = self.P7_1(C7)
        P7_upsample = fluid.layers.resize_nearest(input=P7_x, scale=None, out_shape=(16, 39))
        # P7_upsample = fluid.layers.resize_nearest(input=P7_x, scale=None, out_shape=(10, 10))
        P7_concat = fluid.layers.concat(input=[P8_upsample, P7_x], axis=1)
        P7_x = self.P7_up(P7_concat)
        P7_x = self.bn7_1(P7_x)
        P7_x = self.P7_2(P7_x)
        P7_x = self.bn7_2(P7_x)

        P6_x = self.P6_1(C6)
        P6_upsample = fluid.layers.resize_nearest(input=P6_x, scale=None, out_shape=(32, 77))
        # P6_upsample = fluid.layers.resize_nearest(input=P6_x, scale=None, out_shape=(20, 20))
        P6_concat = fluid.layers.concat(input=[P7_upsample, P6_x], axis=1)
        P6_x = self.P6_up(P6_concat)
        P6_x = self.bn6_1(P6_x)
        P6_x = self.P6_2(P6_x)
        P6_x = self.bn6_2(P6_x)

        P5_x = self.P5_1(C5)
        P5_upsample = fluid.layers.resize_nearest(input=P5_x, scale=None, out_shape=(63, 153))
        # P5_upsample = fluid.layers.resize_nearest(input=P5_x, scale=None, out_shape=(40, 40))
        P5_concat = fluid.layers.concat(input=[P6_upsample, P5_x], axis=1)
        P5_x = self.P5_up(P5_concat)
        P5_x = self.bn5_1(P5_x)
        P5_x = self.P5_2(P5_x)
        P5_x = self.bn5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_upsample = fluid.layers.resize_nearest(input=P4_x, scale=None, out_shape=(125, 306))
        # P4_upsample = fluid.layers.resize_nearest(input=P4_x, scale=None, out_shape=(80, 80))
        P4_concat = fluid.layers.concat(input=[P5_upsample, P4_x], axis=1)
        P4_x = self.P4_up(P4_concat)
        P4_x = self.bn4_1(P4_x)
        P4_x = self.P4_2(P4_x)
        P4_x = self.bn4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_concat = fluid.layers.concat(input=[P4_upsample, P3_x], axis=1)
        P3_x = self.P3_up(P3_concat)
        P3_x = self.bn3_1(P3_x)
        P3_x = self.P3_2(P3_x)
        P3_x = self.bn3_2(P3_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
        """
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        #P5_upsample = fluid.layers.resize_nearest(input=P5_x, scale=None, out_shape=(63, 153))#P4_x.shape[2:]
        P5_upsample = fluid.layers.resize_nearest(input=P5_x, scale=None, out_shape=C4.shape[2:])
        P5_upsample = self.P5_up(P5_upsample)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsample
        #P4_x = self.se5to4(P4_x, P5_upsample)
        P4_upsample = fluid.layers.resize_nearest(input=P4_x, scale=None, out_shape=C3.shape[2:])#, out_shape=(125, 306)
        P4_upsample = self.P4_up(P4_upsample)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsample
        #P3_x = self.se4to3(P3_x, P4_upsample)
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6_1(C5)
        P6_bn = self.bn6(P6_x)

        P7_x = self.P7_1(P6_bn)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]
        """
