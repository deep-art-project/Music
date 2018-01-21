#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 00:10:29 2017

@author: yang
"""

import torch.nn as nn 
import torch
import torch.nn.functional as F
import numpy as np

class wavenet_autoencoder(nn.Module):
    def __init__(self,
        filter_width,
        dilations, 
        residual_channel,
        dilation_channel,
        bottleneck_width,
        quantization_channel,
        deresidual_channel,
        dedilation_channel,
        deskip_channel,
        use_bias):
        super(wavenet_autoencoder,self).__init__()
        self.dilations = dilations
        self.filter_width = filter_width
        self.quantization_channel = quantization_channel	
        self.use_bias = use_bias
        self.bottleneck_width = bottleneck_width
        self.residual_channel = residual_channel
        self.dilation_channel = dilation_channel
        self.deresidual_channel = deresidual_channel
        self.dedilation_channel = dedilation_channel
        self.deskip_channel = deskip_channel
        self._init_causal_layer()
        self._init_conv_layer()
        self._init_deconv_layer()


    def  _init_causal_layer(self):
        self.nc_causal_layer = nn.Conv1d(self.quantization_channel, self.residual_channel, self.filter_width,bias = self.use_bias)
        self.causal_layer = nn.Conv1d(self.quantization_channel,self.deresidual_channel,self.filter_width,bias = self.use_bias)
   
    def _init_conv_layer(self):
        self.dilation_layer_stack = nn.ModuleList()
        self.onedconv_layer_stack = nn.ModuleList()
        for dilation in self.dilations:
            self.dilation_layer_stack.append( nn.Conv1d(self.residual_channel,
                self.dilation_channel,
                self.filter_width, 
                dilation = dilation,bias=self.use_bias))
            self.onedconv_layer_stack.append(nn.Conv1d(self.dilation_channel, self.residual_channel, 1,bias = self.use_bias))
   
    def _init_deconv_layer(self):
        self.dedilation_layer_stack = nn.ModuleList()
        for i,dilation in enumerate(self.dilations):
            current={}
            current['filter_gate'] = nn.Conv1d(self.deresidual_channel,
                                          2*self.dedilation_channel, 
                                          self.filter_width,
                                          dilation =dilation,
                                          bias = self.use_bias)

            current['dense'] =nn.Conv1d(self.dedilation_channel, 
                                        self.deresidual_channel, 
                                        kernel_size =1,
                                        dilation =dilation,
                                        bias=self.use_bias)
            current['skip'] = nn.Conv1d(self.dedilation_channel, 
                                            self.deskip_channel,
                                            dilation =dilation,
                                            kernel_size=1,
                                            bias = self.use_bias)
            self.dedilation_layer_stack.extend(list(current.values()))


    def _encode(self,sample):
        sample = self.nc_causal_layer(sample)
        
        for i,(dilation_layer,ondconv_layer) in enumerate(zip(self.dilation_layer_stack,self.onedconv_layer_stack)):
            current = sample
            sample = F.relu(sample)
            sample = dilation_layer(sample)
            sample = F.relu(sample)
            sample = onedconv_layer(sample)
            _,_,current_length = sample.size()
            current_in_sliced = current[:,:,-current_length]
            sample = sample + current_in_sliced
            
        sample = nn.Conv1d(self.residual_channel, self.bottleneck_width, 1,bias =self.use_bias)    
        sample = nn.AvgPool1d(513,padding = 256,stride =1)
        return sample
    #%% sample is the encoding
    
    def _decode(self,sample,encoding,output_width):
        sample = self.causal_layer(sample)
        skip_contribute_stack = []
        for i, dilation in enumerate(self.dilations):

            [filter_gate_layer,dense_layer,skip_layer]=self.dedilation_layer_stack[4*i:4*i+3]
            
            sample = filter_gate_layer(sample)
            sample = _conditon(sample,encoding)
            _,_,channels = sample.size()
           
            xg = sample[:,:,:channels/2]
            xf = sample[:,:,-channels/2:]
            
            z = F.tanh(xf)*F.sigmoid(xg)

            x_res = dense_layer(z)
            _,_,slice_length = x_res.size()
            x_res_sliced = sample[:,:,-slice_length]
            sample = sample+x_res_sliced
            
            skip = z[:,:,-output_width]
            skip = skip_layer(skip)
            skip_contribute_stack.append(skip)
            
        total = sum(skip)
        
            
        s=F.relu(total)
        s = nn.Conv1d(s,deskip_channel,deskip_channel,1,bias=self.use_bias)
        s= F.relu(total)
        s= nn.Conv1d(s,deskip_channel,quantization_channel,1,bias=self.use_bias)
        batch_size, channels, seq_len = s.size()
        s = s.view(-1, self.quantization_channels)
        s = s.softmax(total)
        return s
            
    def _conditon(x,encoding):
        mb,channels,length = size(encoding)
        mb,channels,length = size(encoding)
        xlength=size(x)[2]
        encoding = encoding.view(mb,channels,length,1)
        x = x.view(mb,channels,length,-1)
        x = x+encoding
        x.view(mb,channels,xlength)
        return x


    def forward(self,wave_sample):
        batch_size, original_channels, seq_len = wave_sample.size()
        output_width = seq_len - self.receptive_field + 1
        encoding = _encode(self,wave_sample)
        s = _decode(self,wave_sample,encoding,output_size)
        return s

net = wavenet_autoencoder(32, [1,2,4,8,16,32,64,128,256,512,
                 1,2,4,8,16,32,64,128,256,512,
                 1,2,4,8,16,32,64,128,256,512,
                 1,2,4,8,16,32,64,128,256,512], 64, 32, 128, 32, 16, 64,32, False)  
print (net.state_dict())  
print(net.parameters)

    
