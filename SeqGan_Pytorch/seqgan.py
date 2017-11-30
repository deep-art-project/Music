# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:03:41 2017

@author: ypc
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
from generator import G_LSTM, Generator
from data_utils import d_data_loader
from rollout import Rollout


def train_epoch(model, dataloader, criterion, optimizer, is_g = True):
    for batch in dataloader:
        x, y = batch
        y = y.type(torch.LongTensor)
        '''if use_gpu:
            x = x.cuda()
            y = y.cuda()'''
        x, y = Variable(x), Variable(y)
        if is_g:
            out, _ = model(x)
            batch_loss = criterion(out.view(-1, out.size(2)), y.view(-1))
        else:
            out = model(x)
            batch_loss = criterion(out, y)
        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
    '''
    running_loss = 0.0
    n_total = 0.0

    for batch in dataloader:
        x, y = batch
        y = y.type(torch.LongTensor)

        mb_size = x.size(0)

        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)
        out, _ = model(x)
        batch_loss = criterion(out.view(-1, out.size(2)), y.view(-1))
        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        running_loss += batch_loss.data[0]
        n_total += mb_size
    return running_loss / n_total'''


def train(n_epoch, model, dataloader, criterion, optimizer, is_g = True):
    for e in range(n_epoch):
        train_epoch(model, dataloader, criterion, optimizer, is_g = is_g)
        '''print('{}/{}'.format(e + 1, n_epoch))
        since = time.time()
        loss = train_epoch(model, dataloader, criterion, optimizer)
        print('Loss: {}, Time: {:.4f}'.format(loss, time.time() - since))
        if (e + 1) % 100 == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(),
                       './checkpoints/model_{}.pth'.format(e + 1))'''

# function of pre-train generator
def pre_train_g(num_epoch_pre_g, fake_lstm, pre_g_data_loader, lr):
	print ('Start pre-training generator...')
	train(num_epoch_pre_g, fake_lstm, pre_g_data_loader,
		nn.CrossEntropyLoss(), optim.Adam(fake_lstm.parameters(), lr = lr))

def pre_train_d(num_batch_pre_neg, num_epoch_pre_d, generator_fake,
                start_token, generated_num, batch_size, discriminator,
                positive_file, negative_file, lr):
	print('Start pre-training discriminator...')
	for _ in range(num_batch_pre_neg):
		generator_fake.generate_samples(start_token, generated_num, negative_file)
		# Produce pre_d_data_loader
		pre_d_data_loader = d_data_loader(positive_file, negative_file, batch_size)
		# Pre-train discriminator
		train(num_epoch_pre_d, discriminator.model, pre_d_data_loader,
        nn.CrossEntropyLoss(), optim.Adam(discriminator.model.parameters(), lr = lr), is_g = False)

#How to define g_loss??????????????????????????????????????????????????????????????????
#def g_loss(y_pre, y):
#zijidingyilosshanshu
'''
class G_func(Function):
    def forward(input, target, rewards):

        result = torch.Tensor()
        return result

    def backward(grad_output):
        return grad_output
'''

class G_loss(nn.Module):
    def __init__(self):
        super(G_loss, self).__init__()
    def forward(self, out, y, rewards):
        '''
        out, y are variables
        '''
        out = out.view(-1, out.size(2))
        y_onehot = Variable(torch.zeros(out.size()).scatter_(1, torch.unsqueeze(y.view(-1).data, dim = 1), 1))
        log_p_hat = torch.sum(y_onehot * torch.log(F.softmax(out)), 1)
        back_loss = torch.sum(log_p_hat * Variable(rewards.contiguous().view(-1)))
        return back_loss




def trian_g(g_steps, generator_fake, start_token, rollout, discriminator, lr):
    loss_func = G_loss()
    optimizer = optim.Adam(generator_fake.model.parameters(), lr = lr)
    for _ in range(g_steps):
        samples = generator_fake.generate(start_token)
        rewards = rollout.get_reward(samples, 16, discriminator)
        x = Variable(samples)
        y = Variable(samples)
        '''if use_gpu:
            x = x.cuda()
            y = y.cuda()'''
        out, _ = generator_fake.model(x)
        back_loss = loss_func(out, y, rewards)

        '''
        # log_p_hat: [batch_size * seq_len, 1]
        out = out.view(-1, out.size(2))

        y_onehot = torch.zeros(out.size()).scatter_(1, torch.unsqueeze(y.view(-1).data, dim = 1), 1)
        log_p_hat = torch.sum(y_onehot * torch.log(F.softmax(out)).data, 1)
        back_loss = Variable(torch.sum(log_p_hat * rewards.contiguous().view(-1)) * torch.ones(1))
        '''
        optimizer.zero_grad()
        back_loss.backward()####youcuo
        nn.utils.clip_grad_norm(generator_fake.model.parameters(), 5)
        optimizer.step()


def train_d(d_steps, k, generator_fake, start_token, generated_num,
            batch_size, discriminator, positive_file, negative_file, lr):
	print('Start training discriminator...')
	# the process of training discriminator is exactly same as pre-training discriminator
	pre_train_d(d_steps, k, generator_fake, start_token, generated_num,
             batch_size, discriminator, positive_file, negative_file, lr)

def train_ad(total_batch, g_steps, d_steps, k, generator_fake, rollout,
             start_token, discriminator, lr, generated_num, batch_size,
             positive_file, negative_file):
    print('Start adversarial training')
    for _ in range(total_batch):
        trian_g(g_steps, generator_fake, start_token, rollout, discriminator, lr)
        rollout = Rollout(generator_fake)
        train_d(d_steps, k, generator_fake, start_token, generated_num,
                batch_size, discriminator, positive_file, negative_file, lr)
