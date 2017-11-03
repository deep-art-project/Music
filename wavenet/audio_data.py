from audio_func import *
from torch.utils.data import Dataset, DataLoader
import os
import glob

class audio_dataset(Dataset):
    def __init__(self,
                 audio_dir,
                 sample_rate,
                 receptive_field,
                 window_length,
                 silence_threshold = None,
                 file_suffix = 'wav',
                 quantization_channels = 256
                 ):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.receptive_field = receptive_field
        self.window_length = window_length
        self.silence_threshold = silence_threshold
        self.file_suffix = file_suffix
        self.quantization_channels = quantization_channels
        self.raw_audio_file_list = self._make_audio_file_list()
    def _make_audio_file_list(self):
        all_files = glob.glob(self.audio_dir + '*.' + self.file_suffix)
        if len(all_files) == 0:
            raise ValueError('No audio file found, please check your \
                             input directory!')
        file_list = []
        for item in all_files:
            item = item.split('/')[-1]
            file_list.append(item)
        return file_list
    def __len__(self):
        return len(self.raw_audio_file_list)
    def __getitem__(self, idx):
        '''
        First read the audio file and perform mu law encode to the
        raw audio file.
        Second pad the left side of encoded audio with padding length
        equals to receptive_field.
        '''
        audio_name = os.path.join(self.audio_dir + \
                                  self.raw_audio_file_list[idx])
        audio = librosa.load(audio_name,
                             sr = self.sample_rate,
                             mono = True)
        audio = audio[0]
        if self.silence_threshold:
            audio = trim_silence(audio, self.silence_threshold)
        audio = np.pad(audio, [[self.receptive_field, 0]], 'constant')
        audio = torch.from_numpy(audio)
        audio = mu_law_encode(audio, self.quantization_channels)

        '''
        Cut the whole audio sequence into pieces of length window_length
        + receptive_field and collect them into a list
        '''
        sample_list = []
        while(len(audio)) > self.receptive_field:
            if len(audio) >= self.receptive_field + self.window_length:
                piece = audio[:(self.receptive_field + \
                                self.window_length - 1)]
                target = audio[self.receptive_field : \
                            (self.receptive_field + self.window_length)]
                sample_list.append({'audio_piece': piece,\
                                'audio_target': target})
                audio = audio[self.window_length:]
            else:
                piece = audio[:-1]
                target = audio[self.receptive_field:]
                audio = audio[self.receptive_field:]
        return sample_list

def audio_data_loader(batch_size, shuffle, num_workers, pin_memory, **kwargs):
    audioDataset = audio_dataset(**kwargs)
    dataloader = DataLoader(audioDataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = num_workers,
                            pin_memory = pin_memory)
    return dataloader

def one_hot_encode(sample_piece,
                   cuda_available = False,
                   quantization_channels = 256):
    '''
    Argument:
        sample_piece:type(dict), format of sample_piece is as follows
                     {'audio_piece':torch.Tensor,\
                      'audio_target':torch.Tensor}
    Return:
        Also a dict with the same key value as input.
        Convert torch.Tensor to one hot encoded torch tensor
    '''
    batch_size, seq_len = sample_piece['audio_piece'].size()
    piece = sample_piece['audio_piece'].view(-1)
    target = sample_piece['audio_target'].view(-1)
    total_len = piece.size()[0]
    piece_one_hot = np.zeros((total_len, quantization_channels))
    piece_one_hot[np.arange(total_len), piece.numpy()] = 1.0
    piece_one_hot = piece_one_hot.reshape(batch_size,
                                          quantization_channels,
                                          seq_len)
    piece_one_hot = torch.FloatTensor(piece_one_hot)
    if cuda_available:
        piece_one_hot = piece_one_hot.cuda()
        target = target.cuda(async = True)
    return piece_one_hot, target

