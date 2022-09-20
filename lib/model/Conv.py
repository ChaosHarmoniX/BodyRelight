import torch
from torch import nn

class MultiConv(nn.Module):
    def __init__(self, filter_channels, is_transpose = False):
        super(MultiConv, self).__init__()
        self.filters = []
        self.is_transpose = is_transpose

        if not is_transpose:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(
                    nn.Conv2d(filter_channels[l], filter_channels[l + 1], kernel_size=4, stride=2)) # 卷积核论文中没有提到大小，应该是4
                self.add_module("conv%d" % l, self.filters[l])
        else:
            # 反卷积块
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(
                    nn.ConvTranspose2d(filter_channels[l], filter_channels[l + 1], kernel_size=4, stride=2)) # 卷积核论文中没有提到大小，应该是4
                self.add_module("deconv%d" % l, self.filters[l])
        
        self.activate_func = torch.nn.functional.relu if is_transpose else torch.nn.functional.leaky_relu
        

    def forward(self, image):
        y = image
        feat_pyramid = [y]
        for i, f in enumerate(self.filters):
            y = f(y)
            if i != len(self.filters) - 1 and i != 0:
                # batch normalization
                y = torch.nn.functional.batch_norm(y)
                # 激活函数
                y = self.activate_func(y)

            # decoder的前三层反卷积的dropout
            if i < 3 and self.is_transpose: # 不清楚是不是应该在激活函数之后
                y = torch.nn.functional.dropout(y, 0.5)

            feat_pyramid.append(y)
        return feat_pyramid
