from mido import MidiFile
from torch.utils.data import Dataset, DataLoader
from utils import tone_to_freq
import numpy as np
import os
import torch


BEGIN_TICK = 0
LENGTH= 1
FREQ = 2
VELOCITY = 3
NUM_SONG_FEATURES = 4


class CRnnGan_Dataset(Dataset):
    '''
    Arguments:
        root_dir:
            The root directory of midi song files. The directory
            structure should be like this:
                /root_dir/: all genre directory
                /one_specific_genre: all composer directory belonging
                                     to this genre
                /one_specific_composer: all songs by this composer

        song_length:
            How many midi events to include in one song piece

        percentage:
            The percentage of whole used to construct this dataset

        output_ticks_per_quarter_note:
            Resolution of the output midi data

    Usage:
        Call function midi_data_loader to get a dataloader.
        e.g.
        >> dataloader = midi_data_loader(**kwargs)
        >> for i_batch, sampled_batch in enumerate(dataloader):
            meta_tensor = sampled_batch["meta_tensor"]
            song_tensor = sample_batch["song_tensor"]
            ......
    '''

    def __init__(
        self,
        root_dir,
        song_length=100,
        percentage=1.0,
        output_ticks_per_quarter_note=384
    ):
        self.root_dir = root_dir
        self.song_length = song_length
        self.percentage = percentage
        self.output_ticks_per_quarter_note = output_ticks_per_quarter_note
        self.filelist, self.genre_list, self.composer_list = \
            self._make_lists()

    def _make_lists(self):
        '''
        Return:
            A list containing all paths of midi song data.
            A list containing all genres.
            A list containing all composers.
        '''
        filelist = []
        genre_list = []
        composer_list = []
        if os.path.exists(self.root_dir) is False:
            raise ("Root midi directory doesn't exist!")
        genres = os.listdir(os.path.join(self.root_dir))
        genre_list.extend(genres)
        for genre in genres:
            genre_dir = os.path.join(self.root_dir, genre)
            composers = os.listdir(genre_dir)
            composer_list.extend(composers)
            for composer in composers:
                composer_dir = os.path.join(genre_dir, composer)
                songs = os.listdir(composer_dir)
                for song in songs:
                    song_path = os.path.join(composer_dir, song)
                    filelist.append(song_path)
        return filelist, genre_list, composer_list

    def __len__(self):
        return int(self.percentage * len(self.filelist))

    def __getitem__(self, idx):
        filepath = self.filelist[idx]
        song_data = self._read_midi(filepath)
        meta_tensor, song_tensor = self._song_data_to_tensor(song_data)
        return {"meta_tensor":meta_tensor,
                "song_tensor":song_tensor}

    def _read_midi(self, filepath):
        if os.path.exists(filepath) is False:
            raise ("Midi file doesn't exist!")
        splited_filepath = filepath.split('/')
        genre, composer = splited_filepath[-3], splited_filepath[-2]
        mid = MidiFile(filepath)
        ticks_per_quarter_note = mid.ticks_per_beat
        input_ticks_per_output_tick = \
            ticks_per_quarter_note / self.output_ticks_per_quarter_note
        # Initialization finished
        # Start converting mid to song data form

        # When converting, remember to ensure the resolution of song data
        # equal to self.output_ticks_per_quarter_note
        song_data = []
        for track in mid.tracks:
            last_event_input_tick = 0
            not_closed_events = []
            for msg in track:
                if (msg.type == 'note_off') or \
                   (msg.type == 'note_on' and msg.velocity == 0):
                    retained_not_closed_events = []
                    for event in not_closed_events:
                        if tone_to_freq(msg.note) == event[FREQ]:
                            abs_tick = msg.time + last_event_input_tick
                            abs_tick = abs_tick / input_ticks_per_output_tick
                            event[LENGTH] = abs_tick - event[BEGIN_TICK]
                            song_data.append(event)
                        else:
                            retained_not_closed_events.append(event)
                    not_closed_events = retained_not_closed_events
                elif msg.type == 'note_on':
                    event = [0.0] * NUM_SONG_FEATURES
                    event[BEGIN_TICK] = last_event_input_tick + msg.time
                    event[BEGIN_TICK] = \
                        event[BEGIN_TICK] / input_ticks_per_output_tick
                    event[FREQ] = tone_to_freq(msg.note)
                    event[VELOCITY] = msg.velocity
                    not_closed_events.append(event)
                last_event_input_tick += msg.time
            for event in not_closed_events:
                event[LENGTH] = self.output_ticks_per_quarter_note
        song_data.sort(key=lambda e: e[BEGIN_TICK])
        return [genre, composer, song_data]

    def _song_data_to_tensor(self, song_data):
        genre, composer, song = song_data
        song = np.asarray(song)
        '''
        First generate one hot embedding of genre and composer.
        '''
        meta_tensor = torch.zeros(len(self.genre_list) +
                                  len(self.composer_list))
        genre_index = self.genre_list.index(genre)
        composer_index = self.composer_list.index(composer)
        composer_index += len(self.genre_list)
        meta_tensor[genre_index] = 1.0
        meta_tensor[composer_index] = 1.0

        '''
        Then generate song_tensor of shape [song_length, NUM_SONG_FEATURES]
        '''
        start_tick = 0
        if len(song) > self.song_length:
            start_tick = np.random.randint(len(song) - self.song_length)
        song_sliced = song[start_tick:(start_tick + self.song_length), :]
        begin_ticks = song_sliced[:, BEGIN_TICK]
        ticks_from_prev_start = np.diff(begin_ticks)
        ticks_from_prev_start = np.concatenate(
            [np.zeros((1)), ticks_from_prev_start]
        )
        song_sliced[:, BEGIN_TICK] = ticks_from_prev_start
        song_tensor = torch.from_numpy(song_sliced)
        song_tensor = song_tensor.float()
        return meta_tensor, song_tensor


def midi_data_loader(batch_size,
                     shuffle,
                     num_workers,
                     pin_memory,
                     **kwargs):
    '''
    Read usage example in the comments of class CRnnGan_Dataset
    '''
    midi_dataset = CRnnGan_Dataset(**kwargs)
    dataloader = DataLoader(
        midi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader

'''
#Following code are for testing
import json


with open('./params/dataset_params.json', 'r') as f:
    params = json.load(f)
f.close()
dataloader = midi_data_loader(**params)
for i, batch in enumerate(dataloader):
    if i >= 2:
        break
    print(batch["meta_tensor"])
    print(batch["song_tensor"])
'''
