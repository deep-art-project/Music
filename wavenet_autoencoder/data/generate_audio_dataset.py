import librosa
import numpy as np
import glob
import math

def trim_silence(audio, threshold, frame_length = 2048):
    '''
    Remove the silence at the beginning and end
    Argument:
        audio:numpy array of size(n)
        threshold:least energy allowed
    '''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length = frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    #indices might be empty if the whole audio is silence
    return audio[indices[0]:indices[-1]] if indices.size \
                                         else audio[0:0]

def divide_one_song(filename,
                    duration,
                    avg_amplitude,
                    threshold):
    print(filename)
    if duration < 20:
        print("Warning duration better be larger than 20")
    total_duration = librosa.get_duration(filename = filename)
    num_pieces = math.ceil(total_duration / duration)
    piece_stack = []
    for num in range(num_pieces):
        audio = librosa.load(filename,
                             sr = 16000,
                             mono = True,
                             offset = num * duration,
                             duration = duration)
        audio = audio[0]
        audio = audio * avg_amplitude / np.mean(np.abs(audio))
        if threshold:
            audio = trim_silence(audio, threshold)
        audio = audio * avg_amplitude / np.mean(np.abs(audio))
        piece_duration = librosa.get_duration(audio, sr = 16000)
        if piece_duration <= 5:
            print('a piece too short or with too much silence!')
        else:
            piece_stack.append(audio)
    if len(piece_stack) == 0:
        raise ValueError('No sound piece generated!')
    else:
        return piece_stack

def piece_name(num):
    l = len(str(num))
    left = 6 - l
    name = str(num)
    for i in range(left):
        name = '0' + name
    return name

def generate_from_songs(audio_dir,
                        to_dir,
                        suffix = '.mp3',
                        duration = 20,
                        avg_amplitude = 0.20,
                        threshold = None
                        ):
    filename_list = glob.glob(audio_dir + '*' + suffix)
    piece_num = 0
    for filename in filename_list:
        piece_stack = divide_one_song(filename,
                                      duration,
                                      avg_amplitude,
                                      threshold)
        for piece in piece_stack:
            piece_num += 1
            piece_path = to_dir + piece_name(piece_num) + '.wav'
            librosa.output.write_wav(piece_path, piece, sr = 16000)

