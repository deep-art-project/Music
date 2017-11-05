from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class Midi_Dataset(Dataset):

    def __init__(self, **kwargs):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def midi_data_loader(batch_size,
                     shuffle,
                     num_workers,
                     pin_memory,
                     **kwargs):
    pass
