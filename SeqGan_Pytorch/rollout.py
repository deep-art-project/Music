# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:03:41 2017

@author: ypc
"""

import numpy as np
import torch


class Rollout(object):
    def __init__(self, generator):
        '''
        Args:
            generator: an instance of class Generator, here it's same as generator
        '''
        self.generator = generator

    def get_reward(self, input_x, rollout_num, discriminator):
        '''
        Args:
            input_x: tensor, [batch_size, seq_length], a batch of negative samples. 
            rollout_num: int, the num of Monte Carlo searching paths.
            discriminator: an instance of class Discriminator.
        Return:
            reward: tensor, [batch_size, seq_length].
        '''
        rewards = []
        # Calculate the reward of each path, and finally take average.
        for i in range(rollout_num):
            # Calculate reward at time t.
            for t in range(1, self.generator.seq_len):
                start_token = input_x[:, :t]
                samples = self.generator.generate(start_token)
                # Calculate the softmax probability of labels of these samples as 1.
                ypred = discriminator.get_pos_prob(samples)
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[t - 1] += ypred

            # Calculate reward at time T.
            ypred = discriminator.get_pos_prob(input_x)
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.generator.seq_len - 1] += ypred
        # Calculate the average of N Monte Carlo search paths.
        rewards = torch.from_numpy(np.transpose(np.array(rewards)) / (1.0 * rollout_num))
        return rewards