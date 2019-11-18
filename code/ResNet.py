import paddle.fluid as fluid
import paddle
from FPN import FPN


class Bottleneck(fluid.dygraph.Layer):
    def __init__(self, name_scope, planes, is_test, stride=1, downsample=None):
        super(Bottleneck, self).__init__(name_scope)
        self.conv1 = fluid.dygraph.Conv2D(name_scope + "_conv1", planes, 1)
        self.bn1 = fluid.dygraph.BatchNorm(name_scope + "_bn1", planes, act="relu", is_test=is_test)

        self.conv2_1 = fluid.dygraph.Conv2D(name_scope + "_conv2_1", planes // 4, 3, stride=stride, padding=1, dilation=1)
        self.conv2_2 = fluid.dygraph.Conv2D(name_scope + "_conv2_2", planes // 4, 3, stride=stride, padding=2, dilation=2)
        self.conv2_3 = fluid.dygraph.Conv2D(name_scope + "_conv2_3", planes // 4, 3, stride=stride, padding=3, dilation=3)
        self.conv2_4 = fluid.dygraph.Conv2D(name_scope + "_conv2_4", planes // 4, 3, stride=stride, padding=4, dilation=4)
        self.bn2 = fluid.dygraph.BatchNorm(name_scope + "_bn2", planes, act="relu", is_test=is_test)

        self.conv3 = fluid.dygraph.Conv2D(name_scope + "_conv3", planes * 4, 1)
        self.bn3 = fluid.dygraph.BatchNorm(name_scope + "_bn3", planes * 4, act="relu", is_test=is_test)

        # self.relu = fluid.layers.relu
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)

        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)
        x_4 = self.conv2_4(x)
        x = fluid.layers.concat(input=[x_1, x_2, x_3, x_4], axis=1)
        x = self.bn2(x)

        x = self.conv3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        return self.bn3(x)


class Make_layer(fluid.dygraph.Layer):
    def __init__(self, name_scope, planes, layernums, is_test, stride=1):
        super(Make_layer, self).__init__(name_scope)
        self.layernums = layernums

        self.downsample = fluid.dygraph.Conv2D(name_scope + "_downsample", planes * 4, 1, stride=stride)

        self.layer1 = Bottleneck(name_scope + "_layer1", planes, stride=stride, is_test=is_test, downsample=self.downsample)
        self.layer2 = Bottleneck(name_scope + "_layer2", planes, is_test=is_test)
        self.layer3 = Bottleneck(name_scope + "_layer3", planes, is_test=is_test)
        # 用for循环 和 [] 存放 layer时，保存权重时会漏掉部分权重
        if (layernums >= 4):  # 因此采用 if ？？？
            self.layer4 = Bottleneck(name_scope + "_layer4", planes, is_test=is_test)
        if (layernums >= 6):
            self.layer5 = Bottleneck(name_scope + "_layer5", planes, is_test=is_test)
            self.layer6 = Bottleneck(name_scope + "_layer6", planes, is_test=is_test)
        """
        self.layers = [Bottleneck(name_scope + "_layer1", planes, stride=stride, is_test=is_test, downsample=self.downsample)]
        for i in range(2, layernums+1): #无法保存权重
            self.layers.append(Bottleneck(name_scope + "_layer%d"%i, planes, is_test=is_test))
        """

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if (self.layernums >= 4):
            x = self.layer4(x)

        if (self.layernums >= 6):
            x = self.layer5(x)
            x = self.layer6(x)
        """
        for layer in self.layers:
            x = layer(x)
        """
        return x


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, is_test=False):
        super(ResNet, self).__init__(name_scope)
        self.conv1 = fluid.dygraph.Conv2D(name_scope + "_conv1", num_filters=64, filter_size=7, stride=2, padding=3, dilation=1)
        self.bn1 = fluid.dygraph.BatchNorm(name_scope + "_bn1", 64, act="relu", is_test=is_test)
        self.maxPooling = fluid.dygraph.Pool2D(name_scope + "maxpooling", pool_size=2, pool_stride=2, pool_type='max')

        self.block1 = Make_layer(name_scope + "_block1", 64, layernums=3, stride=1, is_test=is_test)
        self.block2 = Make_layer(name_scope + "_block2", 128, layernums=4, stride=2, is_test=is_test)
        self.block3 = Make_layer(name_scope + "_block3", 256, layernums=4, stride=2, is_test=is_test)
        self.block4 = Make_layer(name_scope + "_block4", 256, layernums=6, stride=2, is_test=is_test)
        self.block5 = Make_layer(name_scope + "_block5", 256, layernums=6, stride=2, is_test=is_test)
        self.block6 = Make_layer(name_scope + "_block6", 256, layernums=4, stride=2, is_test=is_test)  # 最后一层的输出是带激活函数的
        self.fpn = FPN(name_scope + "_FPN", is_test=is_test)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxPooling(x)
        x = self.block1(x)
        """
        C3 = self.block2(x)
        C4 = self.block3(C3)
        C5 = self.block4(C4)
        return self.fpn([C3, C4, C5])
        """
        C3 = self.block2(x)
        C4 = self.block3(C3)
        C5 = self.block4(C4)
        C6 = self.block5(C5)
        C7 = self.block6(C6)
        return self.fpn([C3, C4, C5, C6, C7])
