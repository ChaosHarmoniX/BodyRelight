import torch
from torch import nn
from lib.model.Conv import *

# TODO: delete
import sys

class ResNetBlock(nn.Module):
    def __init__(self):
        super(ResNetBlock, self).__init__()

        self.conv_block = MultiConv([512, 512, 512]) # input is 512
    
    def forward(self, x):
        print(f'{__file__}:{sys._getframe().f_lineno}: {x.shape}')
        tmp = self.conv_block(x)
        print(f'{__file__}:{sys._getframe().f_lineno}: {tmp.shape}')
        # relu
        return x + tmp
        # print(x.shape) # [512, 2, 3]
        # print(self.conv_block(x).shape) # [512, 6, 6]
        # return x + self.conv_block(x)
        # return self.conv_block(x)