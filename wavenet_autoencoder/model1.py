#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yang
"""

import torch.nn as nn 
import torch
import torch.nn.functional as F
import numpy as np

class wavenet_autoencoder(nn.Module):

    def __init__(self,

        filter_width,
        quantization_channel,
        dilations,
         
        en_residual_channel,
        en_dilation_channel,
       
        en_bottleneck_width,
        en_pool_kernel_size,

       
        de_residual_channel,
        de_dilation_channel,
        de_skip_channel,

        use_bias):

        super(wavenet_autoencoder,self).__init__()

        self.filter_width = filter_width
        self.quantization_channel = quantization_channel
        self.dilations = dilations

        self.en_residual_channel = en_residual_channel
        self.en_dilation_channel = en_dilation_channel
       
        self.en_bottleneck_width = en_bottleneck_width
        self.en_pool_kernel_size = en_pool_kernel_size
        
       
        self.de_residual_channel = de_residual_channel
        self.de_dilation_channel = de_dilation_channel
        self.de_skip_channel = de_skip_channel

        self.use_bias = use_bias

        self.receptive_field = self._calc_receptive_field()
        self.softmax = nn.Softmax(dim=1)

        self._init_encoding()
        self._init_decoding()
        self._init_causal_layer()
        self._init_connection()


    def  _init_causal_layer(self):

        self.en_causal_layer = nn.Conv1d(self.quantization_channel, self.en_residual_channel, self.filter_width,bias = self.use_bias)
        
        self.bottleneck_layer = nn.Conv1d(self.en_residual_channel, self.en_bottleneck_width, 1,bias=self.use_bias)
      
        self.de_causal_layer = nn.Conv1d(self.quantization_channel,self.de_residual_channel,self.filter_width,bias = self.use_bias)
   
    def _calc_receptive_field(self):

        return (self.filter_width - 1) * (sum(self.dilations) + 1) + 1

    def _init_encoding(self):

        self.en_dilation_layer_stack = nn.ModuleList()
        self.en_dense_layer_stack = nn.ModuleList()

        for dilation in self.dilations:

            self.en_dilation_layer_stack.append( nn.Conv1d(

                self.en_residual_channel,
                self.en_dilation_channel,
                self.filter_width, 
                dilation = dilation,
                bias=self.use_bias
                ))

            self.en_dense_layer_stack.append( nn.Conv1d(

                self.en_dilation_channel, 
                self.en_residual_channel, 
                1,
                bias = self.use_bias
                ))
   
    def _init_decoding(self):
        self.de_dilation_layer_stack = nn.ModuleList()

        for i,dilation in enumerate(self.dilations):

            current={}

            current['filter_gate'] = nn.Conv1d(
                self.de_residual_channel,
                2*self.de_dilation_channel, 
                self.filter_width,
                dilation =dilation,
                bias = self.use_bias
                )

            current['dense'] =nn.Conv1d(
                self.de_dilation_channel, 
                self.de_residual_channel, 
                kernel_size =1,
                dilation =dilation,
                bias=self.use_bias
                )

            current['skip'] = nn.Conv1d(
                self.de_dilation_channel, 
                self.de_skip_channel,
                dilation = dilation,
                kernel_size=1,
                bias = self.use_bias
                )

            self.de_dilation_layer_stack.extend(list(current.values()))

    def _init_connection(self):

        self.connection_1 = nn.Conv1d(self.de_skip_channel,self.de_skip_channel,1,bias=self.use_bias)

        self.connection_2 = nn.Conv1d(self.de_skip_channel,self.quantization_channel,1,bias=self.use_bias)


    def _encode(self,sample):
        sample = self.en_causal_layer(sample)
        
        for i,(dilation_layer,dense_layer) in enumerate(zip(self.en_dilation_layer_stack,self.en_dense_layer_stack)):
            
            current = sample

            sample = F.relu(sample)
            sample = dilation_layer(sample)
            sample = F.relu(sample)
            sample = dense_layer(sample)
            _,_,current_length = sample.size()
            current_in_sliced = current[:,:,-current_length:]
            sample = sample + current_in_sliced
            
        sample = self.bottleneck_layer(sample)  
       # print(sample.size())  
        pool1d = nn.AvgPool1d(self.en_pool_kernel_size)
        sample = pool1d(sample)
        return sample
    
    def _decode(self,sample,encoding,output_width):

        current_out = self.de_causal_layer(sample)

        skip_contribute_stack = []

        for i, dilation in enumerate(self.dilations):

            j = 3*i
            current_in  = current_out

            filter_gate_layer,  dense_layer, skip_layer = \
                self.de_dilation_layer_stack[j], \
                self.de_dilation_layer_stack[j+1], \
                self.de_dilation_layer_stack[j+2]

            
            sample = filter_gate_layer(current_in)

       #     en = self._encoding_conv(encoding, self.en_bottleneck_width, 2*self.de_dilation_channel, 1)
            conv1d =  nn.Conv1d(self.en_bottleneck_width, 2*self.de_dilation_channel,1).cuda()
            en = conv1d(encoding)
   #         print(en.size())
    #        print(sample.size())         
           
            sample = self._conditon(sample,en)

          #  print(sample.size())
            _,channels,_ = sample.size()

            xg = sample[:,:-int(channels/2),:]

            xf = sample[:,-int(channels/2):,:]
            
            z = F.tanh(xf)*F.sigmoid(xg)

            x_res = dense_layer(z)

            _,_,slice_length = x_res.size()

            current_in_sliced = current_in[:,:,-slice_length:]
        #    print(sample.size())
         #   print(current_in_sliced.size())
            current_out = current_in_sliced + x_res
            
            skip = z[:,:,-output_width:]

            skip = skip_layer(skip)

            skip_contribute_stack.append(skip)
            
        result = sum(skip_contribute_stack)
        result=F.relu(result)

        
        result = self.connection_1(result)

      #  en = self._encoding_conv(encoding, self.en_bottleneck_width, self.de_skip_channel, 1)
        conv2d =  nn.Conv1d(self.en_bottleneck_width, self.de_skip_channel,1).cuda()
        en = conv2d(encoding)

        result = self._conditon(result,en)
        result= F.relu(result)
        result = self.connection_2(result)
        batch_size, channels, seq_len = result.size()
        result = result.view(-1, self.quantization_channel)
        result = self.softmax(result)
        return result
            
    def _conditon(self,x,encoding):

        mb,channels,length = encoding.size()
        
        xlength=x.size()[2]

        if xlength %length ==0:
            encoding = encoding.view(mb,channels,length,1)
		
            x = x.view(mb,channels,length,-1)

            x = x + encoding

            x= x.view(mb,channels,xlength)
        else:
            repeat_num = int(np.floor(xlength/length))
            encoding_repeat=encoding.repeat(1,1,repeat_num)
            encoding_repeat=torch.cat((encoding_repeat,encoding[:,:,:xlength%length]),2)
            x = x + encoding_repeat
            del encoding_repeat
        return x

    def _encoding_conv(self,encoding,channel_in,channel_out,kernel_size):

        conv1d = nn.Conv1d(channel_in, channel_out,kernel_size)
        en = conv1d(encoding)

        return en

    def forward(self,wave_sample):
#        print(wave_sample.size())
        batch_size, original_channels, seq_len = wave_sample.size()

        output_width = seq_len - self.receptive_field + 1
        print(self.receptive_field)
        encoding = self._encode(wave_sample)
      #  print(type(encoding))
       # encoding = encoding.cuda()
       # print(type(encoding))
        result = self._decode(wave_sample,encoding,output_width)

        return result
if __name__ == '__main__':	
	import json
	with open('./model_params.json','r') as f :
    		params = json.load(f)

	net= wavenet_autoencoder(**params)#filter_width, dilations, dilation_channels, residual_channels, skip_channels, quantization_channels, use_bias
	from torch.autograd import Variable
	input = Variable(torch.randn(1,256,4094))
	output = net(input)
	print(output.size())
	print(torch.sum(output,1))

    
