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
    net = BodyRelightNet(opt).to(device=cuda)
    
    net.load_state_dict(torch.load(f'{opt.checkpoints_path}/{opt.name}/net_epoch_50', map_location=cuda))
    net.eval()

    
    with torch.no_grad():
        image = cv2.imread(os.path.join(eval_opt.eval_input, 'IMAGE.jpg'))
        image = image / 255.0
        
        mask = cv2.imread(os.path.join(eval_opt.eval_input, 'MASK.png'))
        mask = mask[:, :, 0] != 0
        
        image = torch.Tensor(image).T.unsqueeze(0)
        image = image.to(device=cuda)
        
        albedo = cv2.imread(os.path.join(eval_opt.eval_input, 'ALBEDO.jpg'))
        albedo = albedo / 255.0
        for i in range(albedo.shape[0]): # mask albedo
            for j in range(albedo.shape[1]):
                if not mask[i][j]:
                    albedo[i][j] = [0, 0, 0]

        light_dir = os.path.join(opt.dataroot, '..', 'datas', 'sh')
        lights = np.load(os.path.join(light_dir, os.listdir(light_dir)[0]))
        light = lights[eval_opt.eval_light]

        transport = []
        for i in range(9):
            transport_path = os.path.join(eval_opt.eval_input, '%01d.jpg' % (i))
            tmp = cv2.imread(transport_path)[:, :, 0:1] # TODO: 进一步cvt
            if len(transport) == 0:
                transport = tmp
            else:
                transport = np.concatenate((transport, tmp), axis= 2)

        for i in range(transport.shape[0]): # mask transport
            for j in range(transport.shape[1]):
                if not mask[i][j]:
                    transport[i][j] = [0] * 9
        
        cv2.imwrite('./eval/gt_without_reshape.jpg', albedo * (transport @ light))
        
        mask = torch.Tensor(mask.reshape((-1))).T.unsqueeze(0)
        mask = mask.to(device=cuda)
        albedo = torch.Tensor(albedo.reshape((-1, 3))).T.unsqueeze(0).to(cuda)
        light = torch.Tensor(light).T.unsqueeze(0).to(cuda)
        transport = torch.Tensor(transport.reshape((-1, 9))).T.unsqueeze(0).to(cuda)
        image_gt = (albedo * torch.bmm(light, transport)).squeeze(0).reshape((-1, 512, 512)).permute(1, 2, 0).to('cpu')
        cv2.imwrite('./eval/gt.jpg', image_gt.numpy())

        albedo_eval, light_eval, transport_eval = net(image)
        
        error = calc_loss(mask, image, albedo_eval, light_eval, transport_eval, albedo, light, transport, loss).item()
        
        mask = mask.reshape((-1, 1, 512, 512))
        
        for i in range(3):
            albedo_eval[:, 0, :, :] = albedo_eval[:, i, :, :] * mask[:, 0, :, :]
        
        for i in range(9):
            transport_eval[:, i, :, :] = transport_eval[:, i, :, :] * mask[:, 0, :, :]

        light_eval = light_eval.reshape((-1, 3, 9))        
        albedo_eval = albedo_eval.reshape((albedo_eval.shape[0], albedo_eval.shape[1], -1))
        transport_eval = transport_eval.reshape((transport_eval.shape[0], transport_eval.shape[1], -1))
        
        image_eval = albedo_eval * torch.bmm(light_eval, transport_eval) # 因为light_eval和transport_eval的维度是颠倒的，所以矩阵乘法也颠倒一下
        image_eval = image_eval.squeeze(0).reshape((-1, 512, 512)).permute(1, 2, 0).to('cpu')
    cv2.imwrite('./eval/eval_albedo.jpg', albedo_eval.squeeze(0).reshape((-1, 512, 512)).permute(1, 2, 0).to('cpu').numpy()*255)
    print(f'error: {error}')
    cv2.imwrite(eval_opt.eval_output, image_eval.numpy())
