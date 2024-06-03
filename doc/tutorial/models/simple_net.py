# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2021 - 2023
------------------------------------------------------------------------------------------------------------------------
@Author: Diego Gigena Ivanovich - diego.gigena-ivanovich@silicon-austria.com
@File:   nn_models.py
@Time:   2/27/2023 - 10:26 PM
@IDE:    PyCharm
@desc:
------------------------------------------------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x