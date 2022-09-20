import torch
from torch import nn
from Conv import *

class ResNetBlock(nn.Module):
    def __init__(self):
        super(ResNetBlock, self).__init__()

        self.conv_block = MultiConv([512, 512])
    
    def forward(self, x):
        return x + self.conv_block(x)