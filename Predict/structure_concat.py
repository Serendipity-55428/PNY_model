#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: model_concat
@time: 2020/3/27 3:01 下午
'''
import torch
import numpy as np
from Net.structure_cl import ClassiferNet
from Net.structure_re import RegressionNet

class ConcatNet:
    def __init__(self, cl_model_path, re_model_path, test_dataset):
        self._cl_model_path = cl_model_path
        self._re_model_path = re_model_path
        self._test_dataset = torch.from_numpy(test_dataset)
        self.interval_map = {}
        self._acc_TH = 0.5

    def _net_stack(self):
        dev = torch.device(device='cuda:0') if torch.cuda.is_available() else torch.device(device='cpu')
        net_cl = ClassiferNet()
        net_re = RegressionNet()
        net_cl.load_state_dict(state_dict=torch.load(self._cl_model_path, map_location=dev))
        net_re.load_state_dict(state_dict=torch.load(self._re_model_path, map_location=dev))
        self._test_dataset = self._test_dataset.to(dev=dev)
        x, y = self._test_dataset[:, :-1], self._test_dataset[:, -1].reshape(-1, 1)
        result_cl = net_cl(x) #改成区间映射
        result_re = net_re(x)
        result= result_cl + result_re
        acc = torch.where(torch.abs(result-y)<self._acc_TH,
                          torch.tensor(1.), torch.tensor(2.)).sum(dtype=torch.float32) / self._test_dataset.size()[0]
        print("模型总准确率为: %s" % acc)

if __name__ == '__main__':
    pass