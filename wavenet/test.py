from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from model import wavenet
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

class simple_dataset(Dataset):
    def __init__(self):
        self.target = np.random.randint(256, size=2790700)
        self.data = np.random.randn(256, 3200000)
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        sample = {}
        sample["feature"] = torch.FloatTensor(self.data[:, (idx*32000):((idx+1)*32000)])
        sample["target"] = torch.LongTensor(self.target[(idx*27907):((idx+1)*27907)])
        return sample

def test():
    torch.backends.cudnn.benchmark = True
    with open("../wavenet/params/wavenet_params.json", 'r') as f:
        params = json.load(f)
    f.close()
    net = wavenet(**params)
    net = torch.nn.DataParallel(net, device_ids = [0,1])
    net = net.cuda()

    #define dataloader
    dataset = simple_dataset()
    dataloader = DataLoader(dataset, shuffle=True, num_workers=4,
                            batch_size=2, pin_memory=True)

    #define loss function
    loss_func = nn.CrossEntropyLoss().cuda()

    #define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    #train
    for epoch in range(100):
        loss = 0
        forward_time = 0
        backward_time = 0
        optimize_time = 0
        for i, sample in enumerate(dataloader):
            optimizer.zero_grad()
            feature, label = Variable(sample["feature"].cuda(async=True)),\
                             Variable(sample["target"].cuda(async=True))

            now = time.time()
            logits = net(feature).view(-1,256)
            forward_time += (time.time() - now)

            now = time.time()
            loss = loss_func(logits, label.view(-1))
            loss.backward()
            backward_time += (time.time() - now)

            now = time.time()
            optimizer.step()
            optimize_time += (time.time() - now)
        total_time = forward_time + backward_time + optimize_time
        print("Forward consumption is {}".format(forward_time / total_time))
        print("Backward consumption is {}".format(backward_time / total_time))
        print("Optimize consumption is {}".format(optimize_time / total_time))
test()

