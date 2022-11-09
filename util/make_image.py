'''
    Reconstruct the image from the given data.
    This function is used to reconstruct the image from the data
    to validate the data.
'''
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
from lib.model.BodyRelightNet import BodyRelightNet
from lib.options import *
import cv2
import numpy as np
import random

image_path="data/0000/"
light_path="datas/sh/env_sh.npy"

if __name__ == "__main__":
    MASK_MAP=cv2.imread(os.path.join(image_path,"MASK","MASK.png"))
    MASK_MAP=MASK_MAP[:,:,0]!=0
    ALBEDO_MAP=cv2.imread(os.path.join(image_path,"ALBEDO","ALBEDO.jpg"))
    ALBEDO_MAP=ALBEDO_MAP*MASK_MAP[:,:,np.newaxis]/255.0
    
    LIGHT=np.load(light_path)# shape:240,9,3
    TRANSPORT_MAP=os.path.join(image_path,"TRANSFORM")
    transport = []
    for i in range(9):
        transport_path = os.path.join(TRANSPORT_MAP, '%01d.jpg' % (i))
        tmp = cv2.imread(transport_path,)[:, :, 0:1] # TODO: 进一步cvt
        if len(transport) == 0:
            transport = tmp
        else:
            transport = np.concatenate((transport, tmp), axis= 2)
    # GT
    LIGHT=LIGHT[random.randint(0,239)]
    # image = albedo * (transport @ light)
    IMAGE=ALBEDO_MAP*(transport@LIGHT)
    cv2.imwrite("IMAGE.jpg",IMAGE)
    
    