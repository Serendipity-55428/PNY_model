#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: structure
@time: 2020/3/10 12:09 下午
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import cuda
import copy
from functools import reduce
import pickle
import os

class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, flag):
        super().__init__()
        self._flag = flag
        self._conv_style1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self._conv_style2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=1),
            nn.BatchNorm2d(out_channels*2)
        )

    def forward(self, input:torch.Tensor):
        """
        前向传播，分为2层结构和3层结构
        :param input: 输入Tensor，部署在cuda上的
        :return: 网络输出Tensor
        """
        x = self._conv_style1(input=input) if self._flag == 2 else self._conv_style2(input=input)
        x = x + nn.Conv2d(in_channels=input.size()[1], out_channels=x.size()[1], kernel_size=1) \
            if x.size()[1] != input.size()[1] else x + input
        return F.relu(x)

class ClassiferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2) #10*10*32
        self._resnet1 = Resnet(in_channels=32, out_channels=128, kernel_size=3, flag=2) #10*10*128
        self._resnet2 = Resnet(in_channels=32, out_channels=256, kernel_size=3, flag=2) #10*10*256
        # self._resnet3 = Resnet(out_channels=128, kernel_size=3, flag=3)
        self._fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=100), #in_features改
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=25),
            nn.Softmax(dim=1)
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, input:torch.Tensor):
        """
        前向传播
        :param input: dim=1是前4个特征和后续所有特征组合
        :return:
        """
        x = copy.deepcopy(input[:, 4:])
        x = x.reshape(shape=(-1, 1, 10, 10))
        x = self._conv1(x) #-1*32*10*10
        x = F.max_pool2d(x, (2, 2), padding=1) #-1*32*7*7
        for _ in range(2):
            x = self._resnet1(x) #-1*128*7*7
        x = self._resnet2(x) #-1*256*7*7
        x.reshape(-1, reduce(lambda x,y: x*y, x.size[1:]))
        x = torch.cat(tensors=(x, input[:, :4]), dim=1)
        # print(x.size())
        return self._softmax(self._fc(x)) #改全连接层输入特征数量

class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        assert os.path.exists(data_path), '文件路径不存在,请重新检查!'
        with open(file=data_path, mode='rb') as file:
            self._data = pickle.load(file)
    def __len__(self):
        return self._data.shape[0]
    def __getitem__(self, item):
        return self._data[item]
    def __str__(self):
        return '数据集大小为: {0}, 特征向量长: {1}'.format(*self._data)

def weight_init(mod):
    # classname = mod.__class__.__name__
    mod.weight.data = torch.nn.init.xavier_normal_(tensor=mod.weight.data)

def fit():
    data_path = ''
    data_path_test = ''
    model_save_path = ''
    dataloader = DataLoader(dataset=MyDataset(data_path=data_path), batch_size=500, shuffle=True, num_workers=5)
    net = ClassiferNet()
    net.apply(fn=weight_init)
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device=dev)
    with open(file=data_path_test, mode='rb') as file_test:
        data_test = pickle.load(file_test)
    data_test = torch.from_numpy(data_test).to(device=dev)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()
    #training
    l = 0
    for t in range(20000):
        for batch_i, data_batch in enumerate(dataloader):
            data_batch = data_batch.to(device=dev)
            x_batch, y_batch = data_batch[:, :-1], data_batch[:, -1].reshape(-1, 1)
            x_batch = x_batch.reshape(-1, 1, 10, 10)
            output = torch.softmax(net(x_batch))
            loss = criterion(output, y_batch)
            l += loss
            if batch_i % 100 == 0:
                print('第%s轮的第%s组批次的平均损失函数值为%s' % (t, batch_i, (l / 100).cpu().detach().numpy()))
                with torch.no_grad:
                    test_x, test_y = data_test[:, :-1], data_test[:, -1].reshape(-1, 1)
                    predict = torch.argmax(net(test_x), dim=1)
                    acc = torch.where((predict == test_y), torch.tensor(1.),
                                      torch.tensor(0.)).sum(dtype=torch.float32) / predict.size()[0]
                    print('测试集准确率为: %s' % acc.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(obj=net.state_dict(), f=model_save_path)



if __name__ == '__main__':
    a = ClassiferNet()
    print(a.__dict__)
    print((next(a.parameters()) == a.__dict__['_modules']['_conv1'].weight).any())