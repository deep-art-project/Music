#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:51:45 2017

@author: ypc
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def highway(input_, size, num_layers = 1, bias = -2.0, f = F.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    Args:
        input_: a tensor or a list of 2D, batch x n, n is the input_size
        size: int, size is output_size, here set input_size = output_size to match the 
              dimensions of input and output
        num_layers: the number of highway layers.
        f: activation function
    """

    input_size = input_.size(1)
    for idx in range(num_layers):
        h = f(nn.Linear(input_size, size)(input_))
        t = torch.sigmoid(nn.Linear(input_size, size)(input_) + bias)
        output = t * h + (1. - t) * input_ 
        input_ = output
    return output


class CNN(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and full connection layer.
    Args:
        sequence_length: int, the length of sequence data.
        num_classes: int, the number of labels.
        vocab_size: int, the size of vocabulary table.
        embedding_size: int
        filter_sizes: a int list whose each element is kernel size.
        num_filters: a int list whose each element corresponds out_channels.
        l2_reg_lambda: folat, the regularization coefficient.
    Input:
        tensor, [batch_size, sequence_length]
    Output:
        tensor, [batch_size, 1]
    """

    def __init__(self, sequence_length,  vocab_size, embedding_size, 
                 filter_sizes, num_filters, num_classes = 2,
                 l2_reg_lambda = 0.0, dropout_prob = 0.5):
        super(CNN, self).__init__()
       
        self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                      embedding_dim = embedding_size)
        
        #Define conv-maxpooling layer
        self.convs = nn.ModuleList()
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            each_conv = nn.Sequential(
                    nn.Conv2d(in_channels = 1,
                            out_channels = num_filter,
                            kernel_size = (filter_size, embedding_size)),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = (sequence_length - filter_size + 1, 1),
                                 stride = 1))
            self.convs.append(each_conv)

        self.final_dim = sum(num_filters)
        
        #Define highway full connection layer
        self.fc = nn.Sequential(
                            nn.Dropout(dropout_prob),
                            nn.Linear(self.final_dim, num_classes))

    def forward(self, input_):
        x_emd = self.embedding(input_)
        x_emd = torch.unsqueeze(x_emd, 1)
        x_pooled = []
        for each_conv in self.convs:
            x_pooled.append(each_conv(x_emd))
        x_pooled = torch.cat(x_pooled, 1).view(-1, self.final_dim)
        output = self.fc(highway(x_pooled, x_pooled.size(1)))
        return output


class Discriminator(object):
    def __init__(self, model):
        self.model = model 
    def get_pos_prob(self, input_x):
        output = F.softmax(self.model(Variable(input_x)))
        pos_prob = np.array([item[1] for item in output.data.numpy()])
        return pos_prob