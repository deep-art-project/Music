# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:03:41 2017

@author: ypc
"""

import numpy as np
import torch
from torch.utils import data


def g_data_loader(pos_data_file, batch_size, shuffle = True):
    with open(pos_data_file, 'r') as f:
        x = [list(map(int, s.strip().split())) for s in list(f.readlines())]
    x = torch.from_numpy(np.array(x))
    y = torch.zeros(x.size())
    for i in range(x.size(0)):
        y[i, :][:-1], y[i, :][-1] = x[i, :][1:], x[i, :][0]
    dataset = data.TensorDataset(data_tensor = x, target_tensor = y)
    data_loader = data.DataLoader(dataset, batch_size, shuffle = shuffle)
    return data_loader

def d_data_loader(pos_data_file, neg_data_file, batch_size, shuffle = True):
	with open(pos_data_file, 'r') as f:
		pos_examples = [list(map(int, s.strip().split())) for s in list(f.readlines())]
	with open(neg_data_file, 'r') as g:
		neg_examples = [list(map(int, s.strip().split())) for s in list(g.readlines())]
	pos_labels = [1 for _ in pos_examples]
	neg_labels = [0 for _ in neg_examples]

	x = torch.from_numpy(np.array(pos_examples + neg_examples))
	y = torch.from_numpy(np.concatenate([pos_labels, neg_labels], 0))
	dataset = data.TensorDataset(data_tensor = x, target_tensor = y)
	data_loader = data.DataLoader(dataset, batch_size, shuffle = shuffle)
	return data_loader