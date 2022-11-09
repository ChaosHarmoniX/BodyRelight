import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
from lib.model.BodyRelightNet import BodyRelightNet
from lib.options import *
import cv2

opt = BaseOptions().parse() # 一些配置，比如batch_size、线程数

if __name__ == '__main__':
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)
    net = BodyRelightNet(opt).to(device=cuda)
    
    net.load_state_dict(torch.load(opt.load_net_checkpoint_path, map_location=cuda))

    image = cv2.imread(opt.eval_input)
    with torch.no_grad():
        eval_image = net(image)
    
    cv2.imwrite(opt.eval_output, eval_image)
