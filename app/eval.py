import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
from lib.model.BodyRelightNet import BodyRelightNet
from lib.options import *
import cv2
from lib.train_util import calc_loss
from lib.loss_util import loss
import numpy as np
import argparse


if __name__ == '__main__':
    opt = BaseOptions().parse() # 一些配置，比如batch_size、线程数
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_input', default='eval/', help='eval input image path')
    parser.add_argument('--eval_output', default='eval/eval.jpg', help='eval output path')
    parser.add_argument('--eval_light', type=int, default=0, help='eval light')
    eval_opt = parser.parse_args()
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)
    net = BodyRelightNet().to(device=cuda)
    
    net.load_state_dict(torch.load(f'{opt.checkpoints_path}/{opt.name}/net_latest', map_location=cuda))
    net.eval()

    
    with torch.no_grad():
        mask = cv2.imread(os.path.join(eval_opt.eval_input, 'MASK.png'), cv2.IMREAD_GRAYSCALE) / 255.0
        mask_3d = mask[:, :, None]
        
        albedo = cv2.imread(os.path.join(eval_opt.eval_input, 'ALBEDO.jpg'), cv2.IMREAD_COLOR) / 255.0
        albedo = albedo * mask_3d
        albedo = torch.from_numpy(albedo.astype(np.float32)).to(device=cuda)

        light_dir = os.path.join(opt.dataroot, '..', 'datas', 'sh')
        lights = np.load(os.path.join(light_dir, os.listdir(light_dir)[0]))
        light = lights[eval_opt.eval_light]
        light = torch.from_numpy(light.astype(np.float32)).to(device=cuda)

        transport = []
        for i in range(9):
            transport_path = os.path.join(eval_opt.eval_input, '%01d.jpg' % (i))
            tmp = cv2.imread(transport_path, cv2.IMREAD_GRAYSCALE) / 255.0
            if len(transport) == 0:
                transport = tmp[:, :, None]
            else:
                transport = np.concatenate((transport, tmp[:, :, None]), axis= 2)
        transport = transport * mask_3d
        transport = torch.from_numpy(transport.astype(np.float32)).to(device=cuda)

        image = albedo * torch.matmul(transport, light)
        image = 2 * image - 1
        image = image.permute(2, 0, 1)[None, :, :, :].to(device=cuda)
        albedo_eval, light_eval, transport_eval = net(image)

        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device=cuda)
        albedo = albedo.permute(2, 0, 1).unsqueeze(0)
        light = light.unsqueeze(0)
        transport = transport.permute(2, 0, 1).unsqueeze(0)
        
        error = calc_loss(mask, image, albedo_eval, light_eval, transport_eval, albedo, light, transport, loss).item()
        
        cv2.imwrite(eval_opt.eval_output, albedo_eval.squeeze().permute(1, 2, 0).to('cpu').numpy() * 255.0)

    print(f'error: {error}')
