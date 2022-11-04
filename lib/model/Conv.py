import imp
import torch
from torch import nn

class MultiConv(nn.Module):
    image_size = 128
    def __init__(self, filter_channels, is_transpose = False, kernel_size = 4, stride = 2, reshape_times = 0):
        super(MultiConv, self).__init__()
        self.filters = []
        self.batch_norms = []
        self.is_transpose = is_transpose

        padding = ((self.image_size-1) * stride + kernel_size - self.image_size) // 2
        if not is_transpose:
            for l in range(0, len(filter_channels) - 1):
                if reshape_times == 0:
                    self.filters.append(
                        nn.Conv2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=padding))
                else:
                    reshape_times -= 1
                    self.filters.append(
                        nn.Conv2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=1))
                self.add_module("conv%d" % l, self.filters[l])
                if l != 0 and l != len(filter_channels) - 2: # batch norm
                    self.batch_norms.append(nn.BatchNorm2d(filter_channels[l + 1]))
                    self.add_module('batch_norm%d' % (l - 1), self.batch_norms[l - 1])

        else:
            # 反卷积块
            # 128 --> 512
            for l in range(0, len(filter_channels) - 1):
                if reshape_times != 0:
                    reshape_times -= 1
                    self.filters.append(
                        nn.ConvTranspose2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=1))
                else:
                    self.filters.append(
                        nn.ConvTranspose2d(filter_channels[l], filter_channels[l + 1], kernel_size=kernel_size, stride=stride, padding=257))
                self.add_module("deconv%d" % l, self.filters[l])
                if l != 0 and l != len(filter_channels) - 2: # batch norm
                    self.batch_norms.append(nn.BatchNorm2d(filter_channels[l + 1]))
                    self.add_module('batch_norm%d' % (l - 1), self.batch_norms[l - 1])
        
        self.activate_func = torch.nn.functional.relu if is_transpose else torch.nn.functional.leaky_relu
        

    def forward(self, image):
        y = image
        # PIFu里的MultiConv是把各次计算结果放到list里，但是根据我的理解，应该只要最后一次的结果就好了
        for i, f in enumerate(self.filters):
            y = f(y)
            if i != len(self.filters) - 1 and i != 0:
                # batch normalization
                y = self.batch_norms[i - 1](y)
                # 激活函数
                y = self.activate_func(y)

            # decoder的前三层反卷积的dropout
            if i < 3 and self.is_transpose: # TODO: 不清楚是不是应该在激活函数之后
                y = torch.nn.functional.dropout(y, 0.5)
        return y
        # y = image
        # feat_pyramid = [y]
        # for i, f in enumerate(self.filters):
        #     y = f(y)
        #     if i != len(self.filters) - 1 and i != 0:
        #         # batch normalization
        #         y = self.batch_norms[i - 1](y)
        #         # 激活函数
        #         y = self.activate_func(y)

        #     # decoder的前三层反卷积的dropout
        #     if i < 3 and self.is_transpose: # TODO: 不清楚是不是应该在激活函数之后
        #         y = torch.nn.functional.dropout(y, 0.5)

        #     feat_pyramid.append(y)
        # feat_pyramid = torch.cat(feat_pyramid, 0)
        # return feat_pyramid
