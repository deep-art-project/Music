from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch


class audio_dataset(Dataset):

    def __init__(self,
                 audio_path,
                 receptive_field,
                 window_length,
                 cuda_available=False,
                 quantization_channels=256):
        self.audio_path = audio_path
        self.receptive_field = receptive_field
        self.window_length = window_length
        self.cuda_available = cuda_available
        self.quantization_channels = quantization_channels
        data = open(self.audio_path, 'rb')
        data = pickle.load(data)
        self.data = self._make_data_pieces(data)

    def _make_data_pieces(self, data):
        data_pieces = []
        for item in data:
            item = torch.from_numpy(item)
            while(len(item)) > self.receptive_field:
                if len(item) >= self.receptive_field + self.window_length:
                    piece = item[:(self.receptive_field +
                                   self.window_length - 1)]
                    target = item[self.receptive_field:
                                  (self.receptive_field + self.window_length)]
                    item = item[self.window_length:]
                else:
                    item = item[self.receptive_field:]
                target = target.long()
                data_pieces.append({'audio_piece': piece,
                                    'audio_target': target})
        return data_pieces

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return one_hot_encode(self.data[idx],
                              self.cuda_available,
                              self.quantization_channels)


def audio_data_loader(batch_size, shuffle, num_workers, pin_memory, **kwargs):
    audioDataset = audio_dataset(**kwargs)
    print("{} pieces in total".format(len(audioDataset)))
    dataloader = DataLoader(audioDataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    return dataloader


def one_hot_encode(sample_piece,
                   cuda_available=False,
                   quantization_channels=256):
    '''
    Argument:
        sample_piece:type(dict), format of sample_piece is as follows
                     {'audio_piece':torch.Tensor,\
                      'audio_target':torch.Tensor}
    Return:
        Also a dict with the same key value as input.
        Convert torch.Tensor to one hot encoded torch tensor
    '''
    piece, target = sample_piece['audio_piece'], sample_piece['audio_target']
    seq_len = piece.size()[0]
    total_len = seq_len
    piece_one_hot = np.zeros((total_len, quantization_channels))
    piece_one_hot[np.arange(total_len), piece.numpy()] = 1.0
    piece_one_hot = piece_one_hot.reshape(
        quantization_channels, seq_len
    )
    piece_one_hot = torch.FloatTensor(piece_one_hot)
    return {"audio_piece": piece_one_hot, "audio_target": target}
