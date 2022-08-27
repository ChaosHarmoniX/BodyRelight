import torch
from torch import nn
from Conv import *
from .ResNetBlock import ResNetBlock

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.res_net = ResNetBlock()
        self.deconv = MultiConv([512, 512, 256, 128, 64, 9], is_transpose= True) # 9 or 3??
    
    def forward(self, x):
        '''
        :return : output of ResNetBlock and Deconv, the ResNetBlock result should be concatenated to output of encoder
        '''
        res_result = self.res_net(x)
        return res_result, x + self.deconv(res_result)
        
    # def forward(self, x):
    #     return self.deconv(x + self.res_net(x))