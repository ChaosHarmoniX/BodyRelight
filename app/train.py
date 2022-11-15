import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from lib.data.RelightDataset import RelightDataset
from lib.model.BodyRelightNet import BodyRelightNet
from lib.model.Conv import *
from lib.loss_util import loss
from lib.train_util import calc_loss
import torch
from torch import nn
from torch.utils.data import DataLoader

import json
from lib.options import *
from tqdm import tqdm

# delete warnings
import warnings
warnings.filterwarnings("ignore")

# for debug
from visdom import Visdom
import numpy as np
import time

# get options
opt = BaseOptions().parse() # 一些配置，比如batch_size、线程数

if opt.plot:
    train_wind = Visdom()
    # # 初始化窗口信息
    train_wind.line([0.], # Y的第一个点的坐标
            [0.], # X的第一个点的坐标
            win = 'train_loss', # 窗口的名称
            opts = dict(title = 'train_loss') # 图像的标例
    )

    train_epoch_wind = Visdom()
    # # 初始化窗口信息
    train_epoch_wind.line([0.], # Y的第一个点的坐标
            [0.], # X的第一个点的坐标
            win = 'train_epoch_loss', # 窗口的名称
            opts = dict(title = 'train_epoch_loss') # 图像的标例
    )

    test_epoch_wind = Visdom()
    # # 初始化窗口信息
    test_epoch_wind.line([0.], # Y的第一个点的坐标
            [0.], # X的第一个点的坐标
            win = 'test_epoch_loss', # 窗口的名称
            opts = dict(title = 'test_epoch_loss') # 图像的标例
    )
# lr = 0.0002
# batch = 60
def train(net, train_loader, test_loader, loss, num_epochs, updater, device, plot):
    """
    :param net:
    :param train_iter: 训练数据集迭代器
    :param loss: 损失函数, loss(y_hat, y)
    :num_epochs: 迭代次数
    :updater: 优化算法, 如torch.optim.SGD()
    """
    print("Net training begin...")

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
    for epoch in range(start_epoch, num_epochs):
        # for epoch in tqdm(range(start_epoch, opt.num_epoch)):
        print(f'Start epoch {epoch}...')
        train_epoch(epoch, net, train_loader, test_loader, loss, updater, device, plot)
    
    print("Net training end...")


""" 训练"""
def train_epoch(epoch, net, train_dataloader, test_dataloader, loss, updater, device, plot):
    """
    :return : train average loss, test average loss
    """
    epoch_start_time = time.time()

    ### train
    print('Start training...')
    net.train() # 设为训练模式
    train_loss = []
    for index, train_data in enumerate(tqdm(train_dataloader)):
        iter_start_time = time.time()

        image = train_data['image'].to(device)
        mask = train_data['mask'].to(device)
        albedo_gt = train_data['albedo'].to(device)
        light_gt = train_data['light'].to(device)
        transport_gt = train_data['transport'].to(device)

        albedo_hat, light_hat, transport_hat = net(image)
        
        l = calc_loss(mask, image, albedo_hat, light_hat, transport_hat, albedo_gt, light_gt, transport_gt, loss)
        
        # plot frequency can be smaller to accelerate
        if plot:
            train_wind.line([l.item()], [index], win = 'train_loss', update = 'append')
        else:
            print(f'index: {index}, loss: {l.item()}')
        
        train_loss.append(l)

        updater.zero_grad()
        l.backward()
        updater.step()

        iter_net_time = time.time()
        eta = ((iter_net_time - epoch_start_time) / (index + 1)) * len(train_dataloader) - (
                iter_net_time - epoch_start_time) # 当前epoch的剩余时间

        # if index % opt.freq_save == 0 and index != 0:
        if index % opt.freq_save == 0 :
            print(
                '\nName: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | netT: {5:.05f}s | ETA: {6:02d}:{7:02d}'.format(
                    opt.name, epoch, index, len(train_dataloader), l.item(),
                    iter_net_time - iter_start_time, int(eta // 60), int(eta - 60 * (eta // 60))))
            torch.save(net.state_dict(), '%s/%s/net_latest' % (opt.checkpoints_path, opt.name))
            # torch.save(net.state_dict(), '%s/%s/net_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
    torch.save(net.state_dict(), '%s/%s/net_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))
        
    

    # test
    print('Start testing...')
    with torch.no_grad():
        net.eval()
        test_loss = []
        for index, test_data in enumerate(tqdm(test_dataloader)):
            image = test_data['image'].to(device)
            mask = test_data['mask'].to(device)
            albedo_gt = test_data['albedo'].to(device)
            light_gt = test_data['light'].to(device)
            transport_gt = test_data['transport'].to(device)

            albedo_hat, light_hat, transport_hat = net(image)
            
            test_loss.append(calc_loss(mask, image, albedo_hat, light_hat, transport_hat, albedo_gt, light_gt, transport_gt, loss))
    
        train_loss = np.average(train_loss).item()
        test_loss = np.average(test_loss).item()

    if plot:
        train_epoch_wind.line([train_loss], [epoch + 1], win = 'test_epoch_loss', update = 'append')
        test_epoch_wind.line([test_loss], [epoch + 1], win = 'test_epoch_loss', update = 'append')
    else:
        print(f'epoch: {epoch}, train loss: {train_loss}')
        print(f'epoch: {epoch}, test loss: {test_loss}')
        


# 数据集的图片为512 * 512
# aligned 3D models(脸朝前，垂直方向大小一致，padding一致（上下padding都为图片的5%）)。
if __name__ == '__main__':
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)
    net = BodyRelightNet(opt).to(device=cuda)
    
    # train dataset
    train_dataset = RelightDataset(opt, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                    num_workers=0, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_dataloader))
    
    # test dataset
    test_dataset = RelightDataset(opt, 'eval')
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                    num_workers=0, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_dataloader))
    
    # loss
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate, weight_decay=0)
    train(net, train_dataloader, test_dataloader, loss, opt.num_epoch, optimizer, cuda, opt.plot)
