# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:46:59 2017

@author: ypc
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class G_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, 
                sequence_length, num_layers_g = 1, drop_prob_g = 0.5):
        super(G_LSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers_g = num_layers_g
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size = emb_dim, dropout = drop_prob_g,
                            hidden_size = hidden_size, num_layers = num_layers_g, 
                            batch_first = True, bidirectional = False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_x, hs = None):
        '''
        Args:
            input_x: tensor, [batch_size, seq_len]
            hidden_state: Variable(may with cuda), (h, c)
        '''
        
        if len(input_x.size()) == 1:
            input_x = torch.unsqueeze(input_x, dim = 1)
        batch_size = input_x.size(0)
        seq_len = input_x.size(1)
        if torch.cuda.is_available():
            hs = hs.cuda()
                
        x_emb = self.embedding(input_x)
        #out: [batch_size, start_token_len, hidden_size]
        out, hidden_state = self.lstm(x_emb, hs) 
        
        out = self.fc(out.contiguous().view(-1, self.hidden_size))
        #final_out: [batch_size, start_token_len, vocab_size]
        final_out = out.view(batch_size, seq_len, -1)
        return final_out, hidden_state


class Generator(object):
    # Generate Samples
    # Model is an instance of a trainable model class.
    def __init__(self, model):
        self.model = model
        self.seq_len = model.sequence_length

    def generate(self, start_token):
        if len(start_token.size()) == 1:
            start_token = torch.unsqueeze(start_token, dim = 1)
        # start_token: tensor, [batch, start_token_len]
        model = self.model.eval()
        model_input = start_token
        
        #if use_gpu:
            #model_input = model_input.cuda()
            
        model_input = Variable(model_input)
        _, init_state = model(model_input, None)
        result = list(torch.split(start_token, 1, dim = 1))
        model_input = torch.unsqueeze(model_input[:, -1], dim = 1)
        for i in range(self.seq_len - start_token.size(1)):
            #out: [batch, 1, vacab_size]
            out, init_state = model(model_input, init_state)
            #pred: [batch, 1] here -1 is correct?????????????????????????????????????????????????????????????????????????
            pred = torch.multinomial(-1 * torch.log(F.softmax(torch.squeeze(out.data, dim = 1))), 1)
            model_input = pred
            #if use_gpu:
                #model_input = model_input.cuda()

            result.append(pred.data)
        #result: tensor, [batch_size, seq_len]
        result = torch.squeeze(torch.stack(result, dim = 1), dim = 2)
        return result

    def generate_samples(self, start_token, generated_num, output_file):
        '''
        Args:
            generated_num: int, the number of samples generated.
            out_file: the basic shape is [batch_size, seq_len], each word 
                      is the corresponding index
        '''

        generated_samples = []
        for _ in range(int(generated_num / start_token.size(0))):
            generated_samples.extend(self.generate(start_token))

        with open(output_file, 'w') as fout:
            for each_line in generated_samples:
                #buffer = ' '.join([x[0] for x in each_line.numpy()]) + '\n'
                buffer = ' '.join(list(map(str, [x for x in each_line.numpy()]))) + '\n'
                fout.write(buffer)
            