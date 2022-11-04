from turtle import forward
from lib.model.Conv import *
import torch
from torch import nn

from .decoder import Decoder

class BodyRelightNet(nn.Module):
    def __init__(self, opt):
        super(BodyRelightNet, self).__init__()

        self.opt = opt
        self.encoder = MultiConv(filter_channels=[3, 64, 128, 256, 512, 512, 512], reshape_times=2)

        self.albedo_decoder = Decoder(output=3)
        self.transport_decoder = Decoder(output=9)
        # self.light_decoder = MultiConv(filter_channels=[1536, 512, 128, 64, 3], reshape_times=0) # TODO: 四层卷积，但文章中没有说具体的，只知道最终输出为27维
        self.light_decoder = MultiConv(filter_channels=[1536, 512, 128, 64, 27], reshape_times=4, kernel_size=11) # TODO: 四层卷积，但文章中没有说具体的，只知道最终输出为27维
        # 改变kernel大小，直接生成[1,27,1,1]再reshape一下是否可行
    def forward(self, x):
        """
        :param x: [C_3, H, W]
        :return albedo_map, light_map, transport_map
        :albedo_map: [C_3, H, W]
        :light_map: [3, 9]
        :transport_map: [9, H, W]
        """
        feature = self.encoder(x)
        
        albedo_feature, albedo_map = self.albedo_decoder(feature)
        transport_feature, transport_map = self.transport_decoder(feature)

        compose_feature = torch.cat([feature, albedo_feature, transport_feature], 1)
        light_map = self.light_decoder(compose_feature)

        return albedo_map, light_map, transport_map

