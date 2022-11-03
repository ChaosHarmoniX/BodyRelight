import imp
import torch
from torch import nn

# TODO: delete
import sys
from GPUtil import showUtilization as gpu_usage

class MultiConv(nn.Module):
    image_size = 512
    def __init__(self, filter_channels, is_transpose = False, kernel_size = 4, stride = 2, padding_mode='same'):
        super(MultiConv, self).__init__()
        self.filters = []
        self.batch_norms = []
        self.is_transpose = is_transpose

        padding = ((self.image_size-1) * stride + kernel_size - self.image_size) // 2 if padding_mode == 'same' else 1
        if not is_transpose:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(
                    nn.Conv2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=padding))
                self.add_module("conv%d" % l, self.filters[l])
                if l != 0 and l != len(filter_channels) - 2: # batch norm
                    self.batch_norms.append(nn.BatchNorm2d(filter_channels[l + 1]))
                    self.add_module('batch_norm%d' % (l - 1), self.batch_norms[l - 1])

        else:
            # 反卷积块
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(
                    nn.ConvTranspose2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=padding))
                self.add_module("deconv%d" % l, self.filters[l])
                if l != 0 and l != len(filter_channels) - 2: # batch norm
                    self.batch_norms.append(nn.BatchNorm2d(filter_channels[l + 1]))
                    self.add_module('batch_norm%d' % (l - 1), self.batch_norms[l - 1])
        
        self.activate_func = torch.nn.functional.relu if is_transpose else torch.nn.functional.leaky_relu
        

    def forward(self, image):
        y = image
        # PIFu里的MultiConv是把各次计算结果放到list里，但是根据我的理解，应该只要最后一次的结果就好了
        # feat_pyramid = [y]
        # print(f'{__file__}:{sys._getframe().f_lineno}: conv forward: {y.shape}')
        for i, f in enumerate(self.filters):
            print(f'{__file__}:{sys._getframe().f_lineno}: before cal: {y.shape}')
            y = f(y)
            print(f'{__file__}:{sys._getframe().f_lineno}: after conv')
            gpu_usage()
            if i != len(self.filters) - 1 and i != 0:
                # batch normalization
                y = self.batch_norms[i - 1](y)
                print(f'{__file__}:{sys._getframe().f_lineno}: after batch norm')
                gpu_usage()
                # 激活函数
                y = self.activate_func(y)
                print(f'{__file__}:{sys._getframe().f_lineno}: after activate')
                gpu_usage()

            # decoder的前三层反卷积的dropout
            if i < 3 and self.is_transpose: # TODO: 不清楚是不是应该在激活函数之后
                y = torch.nn.functional.dropout(y, 0.5)
                print(f'{__file__}:{sys._getframe().f_lineno}: after dropout')
                gpu_usage()

            # feat_pyramid.append(y)
        # feat_pyramid = torch.cat(feat_pyramid, 0)
        # print(feat_pyramid)
        # return feat_pyramid
        return y
        # y = image
        # feat_pyramid = [y]
        # print(f'{__file__}:{sys._getframe().f_lineno}: conv forward: {y.shape}')
        # for i, f in enumerate(self.filters):
        #     print(f'{__file__}:{sys._getframe().f_lineno}: before cal: {y.shape}')
        #     y = f(y)
        #     print(f'{__file__}:{sys._getframe().f_lineno}: after cal:{y.shape}')
        #     if i != len(self.filters) - 1 and i != 0:
        #         # batch normalization
        #         y = self.batch_norms[i - 1](y)
        #         # 激活函数
        #         y = self.activate_func(y)

        #     # decoder的前三层反卷积的dropout
        #     if i < 3 and self.is_transpose: # TODO: 不清楚是不是应该在激活函数之后
        #         y = torch.nn.functional.dropout(y, 0.5)

        #     feat_pyramid.append(y)
        # print(len(feat_pyramid[0][0][0]))
        # feat_pyramid = torch.cat(feat_pyramid, 0)
        # print(f'{__file__}:{sys._getframe().f_lineno}: conv forward done:{feat_pyramid.shape}')
        # return feat_pyramid
