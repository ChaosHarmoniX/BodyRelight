import torch
from torch import nn
from lib.model.Conv import *

class ResNetBlock(nn.Module):
    def __init__(self):
        super(ResNetBlock, self).__init__()

        self.conv_block = MultiConv([512, 512, 512]) # input is 512
    
    def forward(self, x):
        # relu
        return x + self.conv_block(x)