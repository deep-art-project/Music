from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch


class Target(nn.Module):

    def __init__(self, vocab_size, batch_size, embed_dim, hidden_dim,
                 seq_len, start_token):
        super(Target, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.start_token = start_token

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(
            self.embed_dim, self.hidden_dim
        )
        self.fc = nn.Linear(
            self.hidden_dim,
            self.vocab_size
        )
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal(param, std=1.0)

    def init_hidden(self):
        h = Variable(torch.zeros(
            self.batch_size, self.hidden_dim
        ))
        c = Variable(torch.zeros(
            self.batch_size, self.hidden_dim
        ))
        return h, c

    def forward(self, t, x_t, h_t, c_t):
        x_t_embeded = self.embed(x_t)
        h_tp1, c_tp1 = self.recurrent_unit(x_t_embeded, (h_t, c_t))
        logits = self.fc(h_tp1)
        probs = F.softmax(logits, dim=1)
        next_token = Categorical(probs).sample()
        return t + 1, next_token, h_tp1, c_tp1, logits, next_token

def init_vars(net, use_cuda=False):
    h_t, c_t = net.init_hidden()
    x_t = Variable(nn.init.constant(
        torch.LongTensor(net.batch_size), net.start_token
    ))
    vs = [x_t, h_t, c_t]
    if use_cuda:
        for i, v in enumerate(vs):
            v = v.cuda(async=True)
            vs[i] = v
    return vs

def recurrent_func(f_type='pre'):

    if f_type == 'pre':
        def func(net, real_data, use_cuda=False):
            '''
            Initialize some variables and lists
            '''
            x_t, h_t, c_t = init_vars(net, use_cuda)
            seq_len = net.seq_len
            logits_list = []

            '''
            Perform forward process.
            '''
            for t in range(seq_len):
                _, _, h_t, c_t, logits, next_token = net(t, x_t, h_t, c_t)
                x_t = real_data[:, t].contiguous()
                logits_list.append(logits)
            logits_var = torch.stack(logits_list).permute(1, 0, 2)
            return logits_var
        return func

    elif f_type == 'gen':
        def func(net, use_cuda=False):
            '''
            Initialize some variables and lists
            '''
            x_t, h_t, c_t = init_vars(net, use_cuda)
            seq_len = net.seq_len
            gen_token_list = []

            '''
            Perform forward process.
            '''
            for t in range(seq_len):
                _, x_t, h_t, c_t, logits, next_token = net(t, x_t, h_t, c_t)
                gen_token_list.append(x_t)
            gen_token_var = torch.stack(gen_token_list).permute(1, 0)
            return gen_token_var
        return func

def loss_func(net, real_data, use_cuda=False):
    logits = recurrent_func('pre')(net, real_data, use_cuda).contiguous()
    batch_size, seq_len = real_data.size()
    f = nn.CrossEntropyLoss()
    if use_cuda:
        f = f.cuda()
    inputs = logits.view(batch_size * seq_len, -1)
    target = real_data.view(-1)
    loss = f(inputs, target)
    return loss

def generate(net, use_cuda=False):
    return recurrent_func('gen')(net, use_cuda)
