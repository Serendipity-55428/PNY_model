#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: structure_re
@time: 2020/3/13 9:41 上午
'''
import torch
from torch import nn
import numpy as np
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from collections.abc import Iterable, Iterator, Generator
TH = 1e-2 #判断准确率的阈值

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._module1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.MaxPool2d(2, 2), nn.BatchNorm2d(32), #-1*32*5*5
            nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2, 2, padding=1) #-1*64*3*3
        )
        self._lstm = nn.LSTM(input_size=10, hidden_size=1000, num_layers=2)
        self._mudule3 = nn.Sequential(
            nn.Linear(1000, 100), nn.ReLU(), nn.Dropout(), nn.Linear(100, 200), nn.ReLU(), nn.Dropout(),
            nn.Linear(200, 1), nn.ReLU()
        )
    def forward(self, input:torch.Tensor, **kwargs):
        x = self._module1(input=input[:, 4:])
        x = x.permute(dims=(1, 0, 2)) #将代表批次的维度和seq的维度交换
        x_output, _ = self._lstm(x)
        x = x_output[-1, :, :]
        x = x.squeeze(dim=0)
        return self._mudule3(torch.cat(tensors=[x, input[:, :4]], dim=1)) #与前四个特征相接

class Mydataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(file=data_path, mode='rb') as file:
            self._data = pickle.load(file)

    def __len__(self):
        return self._data.shape[0]
    def __getitem__(self, item):
        return self._data[item]
def weight_init(mod):
    # classname = mod.__class__.__name__
    mod.weight.data = torch.nn.init.xavier_normal_(tensor=mod.weight.data)

def fit():
    Th = 0.5 #测试集判断准确率时的范围阈值
    data_path = ''
    data_path_test = ''
    model_save_path = ''
    dataloader = DataLoader(dataset=Mydataset(data_path), batch_size=500, shuffle=True, num_workers=5)
    net = RegressionNet()
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
            data_batch = data_batch.to(dev)
            x_batch, y_batch = data_batch[:, :-1], data_batch[:, -1].reshape(-1, 1)
            x_batch = x_batch.reshape(-1, 1, 10, 10)
            output = net(x_batch)
            loss = criterion(output, y_batch)
            l += loss
            if t % 100 == 0:
                print('第%s轮的第%s组批次的平均损失函数值为%s' % (t, batch_i, (l / 100).cpu().detach().numpy()))
                with torch.no_grad:
                    test_x, test_y = data_test[:, :-1], data_test[:, -1].reshape(-1, 1)
                    predict = net(test_x)
                    acc = torch.where((torch.abs(predict-test_y) <= Th),
                                      torch.tensor(1.), torch.tensor(0.)).sum(dtype=torch.float32) / test_y.size()[0]
                    print('测试集准确率为: %s' % acc.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(obj=net.state_dict(), f=model_save_path)

if __name__ == '__main__':
    r = RegressionNet()
    print(r.state_dict())
    for i in r.state_dict():
        print(r.state_dict()[i].size())