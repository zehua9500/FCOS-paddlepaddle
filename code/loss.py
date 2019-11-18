import paddle.fluid as fluid
import paddle
import numpy as np


class Loss(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Loss, self).__init__(name_scope)
        self.balance_weight = fluid.dygraph.to_variable(
            np.array([2.7522929, 1.7518761, 2.0946424, 2., 2.0477314, 5.510574,
                      3.068133, 2.5326507, 2.5617657, 2.216007, 4.5600915, 1.0199721,
                      1.8173848, 2.6257467, 2.8263023, 2.3100748, 4.33411, 2.4383526,
                      2.236265, 2.1829107], dtype=np.float32))  # /2.8443
        self.balance_weight.stop_gradient = True

    def iou_loss(self, pred, label, cen_mask):
        # def iou_loss(self, pred, label, cen_label):
        # label shape = [b, -1, 4]  l,r,t,b
        # print("pred ",pred.shape)
        # print("label ",label.shape)
        i_h = fluid.layers.elementwise_min(pred[:, :, 0], label[:, :, 0]) + fluid.layers.elementwise_min(pred[:, :, 1],
                                                                                                         label[:, :, 1])
        i_w = fluid.layers.elementwise_min(pred[:, :, 2], label[:, :, 2]) + fluid.layers.elementwise_min(pred[:, :, 3],
                                                                                                         label[:, :, 3])
        i_area = fluid.layers.elementwise_mul(i_h, i_w)
        # print(i_area)
        # u_area = fluid.layers.elementwise_mul(pred[:, :, 0] + pred[:, :, 1], pred[:, :, 2] + pred[:, :, 3]) + \
        #         fluid.layers.elementwise_mul(label[:, :, 0] + label[:, :, 1], label[:, :, 2] + label[:, :, 3])
        u_area = ((pred[:, :, 0] + pred[:, :, 1]) * (pred[:, :, 2] + pred[:, :, 3])) + ((label[:, :, 0] + label[:, :, 1]) * (label[:, :, 2] + label[:, :, 3]))
        iou = i_area / (u_area - i_area + 1e-7)
        # mask = fluid.layers.greater_than(label, self.limit)
        # mask = paddle.fluid.layers.cast(mask, dtype = "float32")
        # loss = (1 - iou) * cen_mask
        loss = -1 * fluid.layers.log(iou + 1e-7) * cen_mask
        # if(np.isnan(np.sum(loss.numpy()))):
        #    print("pred ", pred)
        return fluid.layers.reduce_sum(loss) / (fluid.layers.reduce_sum(cen_mask) + 1e-5)

    # loss = fluid.layers.elementwise_mul((1 - iou), 10 * cen_label, axis=0)
    # return fluid.layers.reduce_sum(loss) / (fluid.layers.reduce_sum(10 * cen_label) + 1e-5)

    def dice_loss(self, pred, label, cen_label):
        smooth = 1e-5
        cen_label = 5 * cen_label + 1
        i_pos = fluid.layers.elementwise_mul(pred * label, cen_label, axis=0)
        u_pos = fluid.layers.elementwise_mul(label, cen_label, axis=0) + fluid.layers.elementwise_mul(pred, cen_label, axis=0)
        dice_coeff_pos = (fluid.layers.reduce_sum(i_pos) + smooth) / (fluid.layers.reduce_sum(u_pos) + smooth)

        # i_neg = (1 - pred) * (1 - label)
        # u_neg = 2 - pred - label
        # dice_coeff_neg = (fluid.layers.reduce_sum(i_neg) + smooth) /(fluid.layers.reduce_sum(u_neg) + smooth)
        return 1 - dice_coeff_pos * 2  # - dice_coeff_neg * 0.05

    def centerness_loss(self, pred, label, cen_mask, mask):
        # label shape  = [b, -1]
        # 采用 BCE
        eps = 1e-7
        # gamma = 2.0
        pred = fluid.layers.clip(pred, eps, 1 - eps)
        loss = -1 * (label * fluid.layers.log(pred) + (1 - label) * fluid.layers.log(1 - pred))
        # focal_weight = fluid.layers.pow((label - pred), gamma)
        focal_weight = (label - pred) * (label - pred)
        loss = loss * focal_weight
        loss = fluid.layers.elementwise_mul(loss, 10 * cen_mask + 0.3)
        loss = fluid.layers.elementwise_mul(loss, mask)
        # loss = fluid.layers.pow(diff, factor=4.0)
        # loss = paddle.fluid.layers.abs(pred - label)
        # loss = 5*reg_mask*loss + 0.5*(1-reg_mask)*loss
        # loss = paddle.fluid.layers.cast(loss, dtype = "float32")
        # loss = fluid.layers.clip(loss,1e-4, 1.0)
        # sum_loss = fluid.layers.reduce_sum(loss).numpy()
        # print("centerness_loss sum = %s, is_nan = %s"%(sum_loss, np.isnan(sum_loss)))
        return fluid.layers.reduce_mean(loss) * 30

    def focal_loss(self, pred, label, cen_label, mask):
        # cls loss shape = [b,-1, clsNum]
        # label = (1 - label) * 0.005 + label * 0.99
        alpha = 0.5
        gamma = 2.0
        eps = 1e-7
        # print("pred shape ", pred.shape)
        # print("label shape ", label.shape)
        # print("balance_weight shape ", self.balance_weight.shape)
        cls_loss = -1 * (
                alpha * label * fluid.layers.log(pred + eps) * fluid.layers.pow((1 - pred), gamma) + \
                (1 - alpha) * (1 - label) * fluid.layers.log(1 - pred + eps) * fluid.layers.pow(pred, gamma)
        ) * self.balance_weight
        # focal_weight = label * fluid.layers.pow((1 - pred), gamma) + (1 - label) * fluid.layers.pow(pred, gamma)
        # cls_loss = bce_loss * focal_weight
        cls_loss = fluid.layers.elementwise_mul(cls_loss, 10 * cen_label + 0.3, axis=0)
        cls_loss = fluid.layers.elementwise_mul(cls_loss, mask, axis=0)
        return fluid.layers.reduce_mean(cls_loss) * 150  # focal_loss值太少，故把mean换成sum

    def forward(self, cls_out, cls_label, reg_out, reg_label, cent_out, cent_label, cen_mask, mask):
        return self.focal_loss(cls_out, cls_label, cent_label, mask), \
               self.iou_loss(reg_out, reg_label, cen_mask), \
               self.centerness_loss(cent_out, cent_label, cen_mask, mask)  # ,\
        # self.dice_loss(cls_out, cls_label,cent_label)
