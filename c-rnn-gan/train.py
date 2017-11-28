from data import midi_data_loader
from model import Generator, Discriminator
from torch.autograd import Variable
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def get_params(json_dir):
    with open(json_dir, 'r') as f:
        params = json.load(f)
    f.close()
    return params


def get_arguments():
    train_params = get_params('./params/train_params.json')
    crnngan_params = get_params('./params/crnngan_params.json')
    dataset_params = get_params('./params/dataset_params.json')
    return train_params, crnngan_params, dataset_params


def get_optimizer(model, optimizer_type, learning_rate, momentum):
    if optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )

    if optimizer_type == 'rmsprop':
        return optim.RMSprop(
            model.parameters(), lr=learning_rate, momentum=momentum
        )

    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(), lr=learning_rate
        )


def save_model(**kwargs):
    pass


def load_model(**kwargs):
    pass


def train():
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # Get parameters
    train_params, crnngan_params, dataset_params = get_arguments()

    # Get models
    generator = Generator(dataset_params['song_length'], dataset_params['batch_size'], **crnngan_params)

    for epoch in range(train_params['train_epoch']):
        for i, batch in midi_data_loader(**dataset_params):
            song, meta = batch
            song = song.permute(1, 0, 2)

    pass


if __name__ == '__main__':
    train_params, crnngan_params, dataset_params = get_arguments()
    dataloader = midi_data_loader(**dataset_params)
    for i, batch in enumerate(dataloader):
        if i >= 1:
            break
        # print(batch["meta_tensor"])
        print(batch["meta_tensor"])
    pass
