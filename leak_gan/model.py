from scipy.stats import truncnorm
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch

def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated

class Highway(nn.Module):

    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(in_size, out_size)

    def forward(self, x):
        g = F.relu(self.fc1(x))
        t = F.sigmoid(self.fc2(x))
        out = t * g + (1.0 - t) * x
        return out

class Disciminator(nn.Module):

    def __init__(self,
                 seq_len,
                 num_classes,
                 vocab_size,
                 dis_emb_dim,
                 filter_sizes,
                 num_filters,
                 start_token,
                 goal_out_size,
                 step_size,
                 dropout_keep_prob,
                 l2_reg_lambda):
        super(Disciminator, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.start_token = start_token
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)

        self._init_embedding()
        self._init_feature_extractor()
        self.fc = nn.Linear(self.num_filters_total, self.num_classes)
        self.fc.weight.data = truncated_normal(
            self.fc.weight.data.shape
        )
        nn.init.constant(self.fc.bias, 0.1)

    def _init_embedding(self):
        self.embed = nn.Embedding(self.vocab_size + 1, self.dis_emb_dim)
        nn.init.uniform(self.embed.weight, -1.0, 1.0)

    def _init_feature_extractor(self):
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for filter_size, out_channels in \
                zip(self.filter_sizes, self.num_filters):
            current_conv = nn.Conv2d(
                1, out_channels, kernel_size=(filter_size, self.dis_emb_dim)
            )

            '''
            Initialize conv extractor's weight with truncated normal
            '''
            current_conv.weight.data = truncated_normal(
                current_conv.weight.data.shape
            )

            '''
            Initialize conv extractor's bias with constant
            '''
            nn.init.constant(current_conv.bias, 0.1)
            self.convs.append(current_conv)

            current_pool = nn.MaxPool2d((self.seq_len - filter_size + 1, 1))
            self.pools.append(current_pool)
        self.highway = Highway(self.num_filters_total, self.num_filters_total)
        self.dropout = nn.Dropout(1.0 - self.dropout_keep_prob)

    def forward(self, x):
        '''
        Argument:
            x: shape(batch_size * self.seq_len)
               type(Variable containing torch.LongTensor)
        Return:
            pred: shape(batch_size * 2)
                  For each sequence in the mini batch, output the probability
                  of it belonging to positive sample and negative sample.
            feature: shape(batch_size * self.num_filters_total)
                     Corresponding to f_t in original paper
            score: shape(batch_size, self.num_classes)
                   pred = nn.softmax(score)
        '''
        x = self.embed(x)
        batch_size, seq_len, embed_dim = x.data.shape
        x = x.view(batch_size, 1, seq_len, embed_dim)
        pooled_outputs = []
        for conv, pool in zip(self.convs, self.pools):
            h = F.relu(conv(x))
            h = pool(h)
            pooled_outputs.append(h)
        feature = torch.cat(pooled_outputs, dim=1)
        feature = feature.view(-1, self.num_filters_total)
        feature = self.highway(feature)
        feature = self.dropout(feature)
        score = self.fc(feature)
        pred = F.softmax(score)
        return {"pred":pred, "feature":feature, "score":score}
    
    def l2_loss(self):
        W = self.fc.weight
        b = self.fc.bias
        l2_loss = torch.sum(W * W) + torch.sum(b * b)
        l2_loss = self.l2_reg_lambda * l2_loss
        return l2_loss

class Manager(nn.Module):

    def __init__(self, batch_size, hidden_dim, goal_out_size):
        super(Manager, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.recurrent_unit = nn.LSTMCell(
            self.goal_out_size,
            self.hidden_dim
        )
        self.fc = nn.Linear(
            self.hidden_dim,
            self.goal_out_size
        )
        self.goal_init = nn.Parameter(torch.zeros(
            self.batch_size, self.goal_out_size
        ))
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal(std=0.1)
        self.goal_init.data = truncated_normal(
            self.goal_init.data.shape
            )

    def forward(self, f_t, h_m_t, c_m_t):
        h_m_tp1, c_m_tp1 = self.recurrent_unit(f_t, (h_m_t, c_m_t))
        sub_goal = self.fc(h_m_tp1)
        sub_goal = torch.renorm(sub_goal, 2, 0, 1.0)
        return sub_goal, h_m_tp1, c_m_tp1

class Worker(nn.Module):

    def __init__(self,
                 batch_size,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 goal_out_size,
                 goal_size):
        super(Worker, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.goal_size)
        self.goal_change = nn.Parameter(torch.zeros(
            self.goal_out_size, self.goal_size
        ))
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal(param, std=0.1)

    def forward(self, x_t, h_w_t, c_w_t):
        x_t_embeded = self.embedding(x_t) # batch_size * embed_dim
        h_w_tp1, c_w_tp1 = self.recurrent_unit(x_t_embeded, (h_w_t, c_w_t))
        O_tp1 = self.fc(h_w_tp1)
        O_tp1 = O_tp1.view(
            self.batch_size, self.vocab_size, self.goal_size
        )
        return O_tp1, h_w_tp1, c_w_tp1

class Generator(nn.Module):

    def __init__(self, args_dict, step_size):
        super(self, Generator).__init__()
        manager_args = args_dict["manager"]
        worker_args = args_dict["worker"]
        self.step_size = step_size
        self.worker = Worker(**worker_args)
        self.manager = Manager(*manager_args)

    def init_hidden(self):
        h = Variable(torch.zeros(
            self.worker.batch_size, self.worker.hidden_dim
        ))

        c = Variable(torch.zeros(
            self.worker.batch_size, self.worker.hidden_dim
        ))
        return (h, c)

    def forward(self, x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t,
                last_goal, real_goal, t):
        sub_goal, h_m_tp1, c_m_tp1 = self.manager(f_t, h_m_t, c_m_t)
        O, h_w_tp1, c_w_tp1 = self.worker(x_t, h_w_t, c_w_t)
        last_goal_temp = last_goal + sub_goal
        w_t = torch.matmul(
            real_goal, self.worker.goal_change
        )
        w_t = torch.renorm(w_t, 2, 0, 1.0)
        w_t = torch.unsqueeze(w_t, -1)
        logits = torch.squeeze(torch.matmul(O, w_t))
        probs = F.softmax(logits, dim=1)
        x_tp1 = Categorical(probs).sample()
        if (t + 1) % self.step_size == 0:
            return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1,\
                   Variable(torch.zeros(self.batch_size, self.goal_out_size)),\
                   last_goal_temp, t + 1
        else:
            return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1,\
                   last_goal_temp, real_goal, t + 1
