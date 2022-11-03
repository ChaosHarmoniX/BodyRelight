import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from lib.data.RelightDataset import RelightDataset
from lib.model.BodyRelightNet import BodyRelightNet
from lib.model.Conv import *
from lib.model.loss_util import loss
import torch
from torch import nn
from torch.utils.data import DataLoader

import json
from lib.options import *
from tqdm import tqdm

# lr = 0.0002
# batch = 60

# get options
opt = BaseOptions().parse() # 一些配置，比如batch_size、线程数

def train(net, train_loader, loss, num_epochs, updater):
    """
    :param net:
    :param train_iter: 训练数据集迭代器
    :param loss: 损失函数, loss(y_hat, y)
    :num_epochs: 迭代次数
    :updater: 优化算法, 如torch.optim.SGD()
    """
    print("Begin training...")

    # load checkpoints
    if opt.load_net_checkpoint_path is not None:
        print('loading for net ...', opt.load_net_checkpoint_path)
        net.load_state_dict(torch.load(opt.load_net_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/net_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/net_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        net.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in tqdm(range(start_epoch, opt.num_epoch)):
        train_epoch(net, train_loader, loss, updater)
    
    print("End training...")


""" 训练"""
def train_epoch(net, train_dataloader, loss, updater):
    net.train() # 设为训练模式
    
    for train_data in train_dataloader:
        image = train_data['image']
        mask = train_data['mask']
        albedo_gt = train_data['albedo']
        light_gt = train_data['light']
        transport_gt = train_data['transport']

        albedo_hat, light_hat, transport_hat = net(image)
        for i in range(albedo_hat.shape[0]): # mask the output
            if not mask[i]:
                albedo_hat[i] = [0, 0, 0]
                transport_hat[i] = [0] * 9
        
        image_hat = albedo_hat * (transport_hat @ light_hat)
        l = loss(albedo_hat, light_hat, transport_hat, image_hat, albedo_gt, light_gt, transport_gt, image)
        updater.zero_grad()
        l.backward()
        updater.step()

# 数据集的图片为1024 * 1024
# aligned 3D models(脸朝前，垂直方向大小一致，padding一致（上下padding都为图片的5%）)。
if __name__ == '__main__':
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)
    net = BodyRelightNet(opt).to(device=cuda)
    
    train_dataset = RelightDataset(opt, cuda, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                    num_workers=0, pin_memory=opt.pin_memory)
    # loss
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate, weight_decay=0)
    train(net, train_dataloader, loss, opt.num_epoch, optimizer)
