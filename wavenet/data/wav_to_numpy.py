import glob
import librosa
import numpy as np
import pickle


def mu_law_encode(audio, quantization_channels = 256):
    '''
    Arguments:
        audio: type(np.array), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        np.array of shape(sequence length)
    Return:
        np.array with each element ranging from 0 to 255
        The size of return array is the same as input tensor
    '''
    mu = quantization_channels - 1
    safe_audio_abs = np.abs(np.clip(audio, -1.0, 1.0))
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    encoded = (signal + 1) / 2 * mu + 0.5
    return encoded.astype(np.int32)

def main(audio_dir, suffix='.wav'):
    pattern = audio_dir + '*' + suffix
    file_list = glob.glob(pattern)
    audio_list = []
    for item in file_list:
        audio = librosa.load(item, sr=16000, mono=True)
        audio = audio[0]
        encoded = mu_law_encode(audio)
        audio_list.append(encoded)
    output = open(audio_dir + "np_audio.pkl", 'wb')
    pickle.dump(audio_list, output)

main('/data/zyj/piano_pieces/')
