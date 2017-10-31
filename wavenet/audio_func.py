import numpy as np
import torch
import librosa

def mu_law_encode(audio, quantization_channels = 256):
    '''
    Arguments:
        audio: type(torch.FloatTensor), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        torch Tensor of shape(sequence length)
    Return:
        A torch tensor with each element ranging from 0 to 255
        The size of return tensor is the same as input tensor
    '''
    mu = torch.Tensor([quantization_channels - 1])
    mu = mu.float()
    safe_audio_abs = torch.abs(torch.clamp(audio, -1.0, 1.0))
    magnitude = torch.log1p(mu * safe_audio_abs) / torch.log1p(mu)
    signal = torch.sign(audio) * magnitude
    encoded = (signal + 1) / 2 * mu + 0.5
    return encoded.long()

def mu_law_decode(output, quantization_channels = 256):
    '''
    Argument:
        output:quantized values, data type(torch Tensor),
               data shape(sequence_length), each element
               in output is an int ranging from 0 to 255
    Return:
        A torch Tensor with each element's absolute value
        less than 1, the size of returned tensor is same as
        Argument 'output'
    '''
    mu = torch.Tensor([quantization_channels - 1])
    mu = mu.float()
    signal = 2.0 * (output.float() / mu) - 1.0
    magnitude = (1.0 / mu) * ((1.0 + mu) ** torch.abs(signal) - 1.0)
    return torch.sign(signal) * magnitude

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

