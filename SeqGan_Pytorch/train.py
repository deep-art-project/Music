# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:05:02 2017

@author: ypc
"""

import numpy as np
import torch

from data_utils import g_data_loader
from generator import G_LSTM, Generator
from discriminator import CNN, Discriminator
from seqgan import pre_train_g, pre_train_d, train_ad
from rollout import Rollout



#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
Emb_dim = 32 # embedding dimension
Vocab_size = 5000
Hidden_size = 32 # hidden state dimension of lstm cell
Seq_len = 20 # sequence length
Seed = 88
Batch_size = 64
Start_token = torch.from_numpy(np.array([0] * Batch_size))

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################

Positive_file = 'save/real_data.txt'
Negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
Generated_num = 128
use_gpu = False
# =======================================================================================================
'''Pre-training Hyper-parameters'''
Num_epoch_pre_g = 1 
Num_batch_pre_neg = 1
Num_epoch_pre_d = 1
LR = 0.001
Total_batch = 2
G_steps = 1
D_steps = 5
K = 3

# =======================================================================================================
'''Use the oracle(target_lstm) model to provide the positive examples'''
real_lstm = G_LSTM(Vocab_size, Emb_dim, Hidden_size, Seq_len)
generator_real = Generator(real_lstm)
generator_real.generate_samples(Start_token, Generated_num, Positive_file)

# =======================================================================================================
'''Pre-train generator'''

# Randomly initialze the fake generator(generator_fake)
fake_lstm = G_LSTM(Vocab_size, Emb_dim, Hidden_size, Seq_len)
generator_fake = Generator(fake_lstm)
# Produce pre_g_data_loader
pre_g_data_loader = g_data_loader(Positive_file, Batch_size)
# Pre-train generator
pre_train_g(Num_epoch_pre_g, generator_fake.model, pre_g_data_loader, LR)

# =======================================================================================================
'''Pre-train discriminator'''

# Randomly initialze the discriminator
cnn = 	CNN(sequence_length = Seq_len, num_classes = 2, vocab_size = Vocab_size,
            embedding_size = dis_embedding_dim, filter_sizes = dis_filter_sizes, 
            num_filters = dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
discriminator = Discriminator(cnn)

# Pre-train discriminator
pre_train_d(Num_batch_pre_neg, Num_epoch_pre_d, generator_fake, 
            Start_token, Generated_num, Batch_size, discriminator,
            Positive_file, Negative_file, LR)

rollout = Rollout(generator_fake)
# ========================================================================================================
'''Start Adversarial Training'''

train_ad(Total_batch, G_steps, D_steps, K, generator_fake, rollout,
         Start_token, discriminator, LR, Generated_num, Batch_size, 
         Positive_file, Negative_file)

# ========================================================================================================