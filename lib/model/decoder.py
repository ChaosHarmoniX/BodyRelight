import torch
from torch import nn
from lib.model.Conv import *
from .ResNetBlock import ResNetBlock

class Decoder(nn.Module):
    def __init__(self, output):
        """
        :param output: 9 for transport and 3 for albedo
        """
        super(Decoder, self).__init__()

        self.res_net = ResNetBlock()
        self.deconv = MultiConv([512, 512, 256, 128, 64, output], is_transpose= True, reshape_times=2)
    
    def forward(self, x):
        '''
        :return : output of ResNetBlock and Deconv, the ResNetBlock result should be concatenated to output of encoder
        '''
        res_result = self.res_net(x)
        # return res_result, x + self.deconv(res_result) # TODO: 此处可能有误，是根据ResNet的skip connection推测的
        return res_result, self.deconv(x + res_result) # TODO: 也可能不是+而是concat