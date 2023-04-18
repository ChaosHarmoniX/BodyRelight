import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
from lib.model.BodyRelightNet import BodyRelightNet
from lib.options import *
import numpy as np
import cv2

class Relight:
    def __init__(self, opt = BaseOptions().parse()) -> None:
        # set cuda
        self.device = torch.device('cuda:%d' % opt.gpu_id)
        self.net = BodyRelightNet().to(device=self.device)
        
        self.net.load_state_dict(torch.load(f'{opt.checkpoints_path}/{opt.name}/net_latest', map_location=self.device))
        self.net.eval()

    def relight_batch(self, image, mask, light, need_normalize = True):
        """
        Parameters:
        - image: `numpy.ndarray([batch, 1024, 1024, 3])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - mask: `numpy.ndarray([1024, 1024])`, default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - light: `numpy.ndarray([batch, 9, 3])`
        - need_normalize: if True, the input image will be divided by 255 to be normalized to [0, 1] when relighting

        Return:
        - religted_image: `numpy.ndarray([batch, 1024, 1024, 3])`, ranging [0, 255]
        """
        with torch.no_grad():
            if need_normalize:
                image = image / 255.0
                mask = mask / 255.0
            image = image * mask[:, :, :, None]

            image = torch.from_numpy(image).float()
            light = torch.from_numpy(light).float()

            image = image.to(self.device)
            light = light.to(self.device)

            image = 2 * image - 1 # normalize to [-1, 1]
            
            image = image.permute(0, 3, 1, 2) # [batch, 1024, 1024, 3] -> [batch, 3, 1024, 1024]
            albedo_eval, light_eval, transport_eval = self.net(image)
            transport_eval = transport_eval.permute(0, 2, 3, 1) # [batch, 9, 1024, 1024] -> [batch, 1024, 1024, 9]
            # light = light.permute(0, 2, 1) # [batch, 9, 3] -> [batch, 3, 9]
            albedo_eval = albedo_eval.permute(0, 2, 3, 1) # [batch, 3, 1024, 1024] -> [batch, 1024, 1024, 3]
            shading = torch.clamp(torch.einsum('ijkl,ilm->ijkm', transport_eval, light), 0.0, 10.0)
            image_eval = torch.clamp(albedo_eval * shading, 0.0, 1.0)
            
            return image_eval.to('cpu').numpy() * 255.0
    
    def relight(self, image, mask, light, need_normalize = True):
        """
        Parameters:
        - image: numpy.ndarray([1024, 1024, 3]), default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - mask: numpy.ndarray([1024, 1024]), default ranging [0, 255], if need_normalize is False, it should be ranging [0, 1]
        - light: numpy.ndarray([9, 3])
        - need_normalize: if True, the input image will be divided by 255 to be normalized to [0, 1] when relighting

        Return:
        - religted_image: numpy.ndarray([1024, 1024, 3]), ranging [0, 255]
        """
        with torch.no_grad():
            if need_normalize:
                image = image / 255.0
                mask = mask / 255.0
            image = image * mask[:, :, None]

            image = torch.from_numpy(image).float()
            light = torch.from_numpy(light).float()

            image = image.to(self.device)
            light = light.to(self.device)
            
            image = 2 * image - 1 # normalize to [-1, 1]
            image = image.permute(2, 0, 1) # [1024, 1024, 3] -> [3, 1024, 1024]
            image.unsqueeze_(dim = 0)
            albedo_eval, light_eval, transport_eval = self.net(image)

            albedo_eval.squeeze_(dim = 0)
            transport_eval.squeeze_(dim = 0)
            transport_eval = transport_eval.permute(1, 2, 0) # [9, 1024, 1024] -> [1024, 1024, 9]
            shading = torch.clamp(torch.matmul(transport_eval, light), 0.0, 10.0)
            albedo_eval = albedo_eval.permute(1, 2, 0) # [3, 1024, 1024] -> [1024, 1024, 3]
            image_eval = torch.clamp(albedo_eval * shading, 0.0, 1.0)

            return image_eval.to('cpu').numpy() * 255.0
    

if __name__ == '__main__':
    opt = BaseOptions().parse()
    relight = Relight(opt)
    image = cv2.imread('./eval/IMAGE_L.jpg', cv2.IMREAD_COLOR)
    mask = cv2.imread('./eval/MASK.png', cv2.IMREAD_GRAYSCALE)
    light = np.load('./datas/LIGHT/env_sh.npy')[0]

    # --- test non-batch version ---
    # image_eval = relight.relight(image, mask, light)
    # cv2.imwrite('./eval/IMAGE_L_eval.jpg', image_eval)
    # # Don't use cv2.imshow directly, it can't deal with float32 ranging from 0 to 255 properly
    # image_eval = image_eval.astype(np.uint8)
    # cv2.imshow('image_eval', image_eval)
    # cv2.waitKey(0)

    # --- test batch version ---
    image = image[None, :, :, :]
    mask = mask[None, :, :]
    light = light[None, :, :]
    image_eval = relight.relight_batch(image, mask, light)
    image_eval = image_eval[0]
    cv2.imwrite('./eval/IMAGE_L_eval.jpg', image_eval)
    # Don't use cv2.imshow directly, it can't deal with float32 ranging from 0 to 255 properly
    image_eval = image_eval.astype(np.uint8)
    cv2.imshow('image_eval', image_eval)
    cv2.waitKey(0)
