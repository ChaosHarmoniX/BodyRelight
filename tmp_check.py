import torch
import cv2
import os

for i in range(258):
    mask_path = os.path.join('./data', '{0:04d}'.format(i), 'MASK', 'MASK.png')
    mask = cv2.imread(mask_path)
    try:
        mask = mask[:, :, 0] != 0
    except:
        print(f"{mask_path} not exist")