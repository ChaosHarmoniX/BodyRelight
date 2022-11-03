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
        self.deconv = MultiConv([512, 512, 256, 128, 64, output], is_transpose= True)
    
    def forward(self, x):
        '''
        :return : output of ResNetBlock and Deconv, the ResNetBlock result should be concatenated to output of encoder
        '''
        res_result = self.res_net(x)
        # print('decoder')
        # print(res_result.shape) # [512, 2, 2]
        # print(x.shape) # [512, 6, 6]
        # print(self.deconv(res_result).shape) # [3, 126, 126]
        # print('all')
        # return res_result, x + self.deconv(res_result) # TODO: 此处可能有误
        return self.deconv(x + self.res_net(x)) # TODO: 也可能不是+而是concat
        
    # def forward(self, x):