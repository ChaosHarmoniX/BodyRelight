from turtle import forward
from lib.model.Conv import *
import torch
from torch import nn

from .decoder import Decoder

class BodyRelightNet(nn.Module):
    def __init__(self):
        super(BodyRelightNet, self).__init__()

        self.encoder = MultiConv(filter_channels=[64, 128, 256, 512, 512, 512])

        self.albedo_decoder = Decoder()
        self.transport_decoder = Decoder()
        self.light_decoder = MultiConv(filter_channels=[]) # 四层卷积，但文章中没有说具体的，只知道最终输出为27维
    
    def forward(self, x):
        feature = self.encoder(x)
        
        albedo_feature, albedo_map = self.albedo_decoder(feature)
        transport_feature, transport_map = self.transport_decoder(feature)

        compose_feature = torch.cat([feature, albedo_feature, transport_feature])
        light_map = self.light_decoder(compose_feature)

        return albedo_map, light_map, transport_map

