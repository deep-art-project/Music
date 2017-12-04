from model import Disciminator
from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def main(type="discriminator"):
    if type == "discriminator":
        test_discriminator()
    elif type == "manager":
        test_manager()
    elif type == "worker":
        test_worker()
    elif type == "dataloader":
        test_dataloader()
    elif type == "train":
        test_train()
    elif type == "generate":
        test_generate()
    else:
        raise ("Invalid test type!")

def test_discriminator():
    net = Disciminator(
        20, 2, 2000, 64,
        [1, 2, 4, 6, 8, 10, 20],
        [100, 100, 100, 100, 100, 160, 160],
        0, 820, 4, 0.75, 0.2
    )
    print(net)
    sentence = np.random.randint(2000, size=(64, 20))
    sentence = Variable(torch.from_numpy(sentence).long())
    target = np.random.randint(2, size=(64))
    target = Variable(torch.from_numpy(target).long())
    out_dict = net(sentence)
    print("Disciminator forward test passed.")
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(out_dict["score"], target)
    loss.backward()
    print("Disciminator backward test passed.")


def test_manager():
    pass

def test_worker():
    pass

def test_dataloader():
    pass

def test_train():
    pass

def test_generate():
    pass

main()
