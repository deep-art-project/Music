from scipy.stats import truncnorm
import torch.nn as nn
import torch.nn.functional as F
import torch

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
        self.fc.weight.data = self.truncated_normal(
            self.fc.weight.data.shape
        )
        nn.init.constant(self.fc.bias, 0.1)

    def truncated_normal(self, shape, lower=-0.2, upper=0.2):
        size = 1
        for dim in shape:
            size *= dim
        w_truncated = truncnorm.rvs(lower, upper, size=size)
        w_truncated = torch.from_numpy(w_truncated).float()
        w_truncated = w_truncated.view(shape)
        return w_truncated

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
            current_conv.weight.data = self.truncated_normal(
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
