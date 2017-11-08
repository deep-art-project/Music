from audio_func import mu_law_decode
from collections import OrderedDict
from model import wavenet
from torch.autograd import Variable
from train import load_model
import json
import librosa
import os
import torch
import torch.nn.functional as F


def predict_next(net, note, state_queue=None):
    '''
    Arguments:
        net: Pretrained wavenet nodel
        state_queue: A queue that store states previously calculated.
                     The number of states stored in each layer is the
                     same as the dilation number of that layer
        note: The new coming music note in one hot form
    Return:
        new_state_queue: In each layer remove the left most one, add the
                         new state to the right most
        new_note: New generated music note
    Remark:
        An illustration of fast generate algorithm is in the url below.
        https://github.com/tomlepaine/fast-wavenet
    '''
    if state_queue is None:
        assert note.size()[2] == net.receptive_field
        state_queue = OrderedDict()
        current_out = net.causal_layer(note)
        state_queue['causal_layer'] = note[:, :, -1].contiguous()
        state_queue['causal_layer'] = state_queue['causal_layer'].view(
            1,
            net.quantization_channels,
            1
        )
        skip_contribution_stack = []
        for i, dilation in enumerate(net.dilations):
            layer_name = 'block_' + str(i + 1)
            state_queue[layer_name] = current_out[:, :, -dilation:].contiguous()
            state_queue[layer_name] = state_queue[layer_name].view(
                1,
                net.residual_channels,
                dilation
            )
            current_in = current_out
            j = 4 * i
            filter_layer, gate_layer, dense_layer, skip_layer = \
                net.dilation_layer_stack[j], \
                net.dilation_layer_stack[j + 1],\
                net.dilation_layer_stack[j + 2],\
                net.dilation_layer_stack[j + 3]
            current_filter = filter_layer(current_in)
            current_gate = gate_layer(current_in)
            combined = F.sigmoid(current_gate) * F.tanh(current_filter)
            current_dense = dense_layer(combined)
            _, _, current_len = current_dense.size()
            current_in_sliced = current_in[:, :, -current_len:]
            current_out = current_dense + current_in_sliced

            skip = combined[:, :, -1:]
            skip = skip_layer(skip)
            skip_contribution_stack.append(skip)
    else:
        assert note.size()[2] == 1
        '''
        Define a util function to perform calculation in one layer.
        '''
        def one_layer_forward(state, note, layer_list, layer_type='causal'):
            batch_size, channels, seq_len = state.size()
            layer_input = torch.ones(batch_size, channels, seq_len + 1)
            layer_input[:, :, :-1] = state.data
            layer_input[:, :, -1] = note.data
            layer_input = Variable(layer_input)
            if layer_type == 'causal':
                layer = layer_list[0]
                out = layer(layer_input)
                return out
            elif layer_type == 'res_block':
                filter_layer, gate_layer, dense_layer, skip_layer = \
                    layer_list[0], layer_list[1], layer_list[2], layer_list[3]
                current_filter = filter_layer(layer_input)
                current_gate = gate_layer(layer_input)
                combined = F.sigmoid(current_gate) * F.tanh(current_filter)
                current_dense = dense_layer(combined)
                _, _, current_len = current_dense.size()
                layer_input_sliced = layer_input[:, :, -current_len:]
                current_out = current_dense + layer_input_sliced

                # Then calculate the skip contributions to form the prediction
                skip = combined[:, :, -1:]
                skip = skip_layer(skip)
                return current_out, skip
        '''
        Define a util funciton to update state queue in one layer
        '''
        def one_layer_update(state, note):
            new_state = torch.ones(state.size())
            if state.size()[2] > 1:
                new_state[:, :, :-1] = state.data[:, :, 1:]
            new_state[:, :, -1] = note.data
            return Variable(new_state)


        '''
        Perform network forward process and update state queue at the
        same time.
        '''
        causal_state = state_queue['causal_layer']
        layer_list = [net.causal_layer]
        note_out = one_layer_forward(causal_state,
                                     note,
                                     layer_list)
        state_queue['causal_layer'] = one_layer_update(causal_state, note)
        skip_contribution_stack = []
        for i, dilation in enumerate(net.dilations):
            note_in = note_out
            block_state = state_queue["block_" + str(i + 1)]
            j = 4 * i
            layer_list = [net.dilation_layer_stack[j + k] for k in range(4)]
            note_out, skip = one_layer_forward(block_state,
                                               note_in,
                                               layer_list,
                                               'res_block')
            skip_contribution_stack.append(skip)
            state_queue["block_" + str(i + 1)] = one_layer_update(block_state,
                                                                  note_out)
    total = sum(skip_contribution_stack)
    total = F.relu(total)
    total = net.post_process_1(total)
    total = F.relu(total)
    total = net.post_process_2(total)

    # Finally, for each row in total, perform a softmax
    # to get probability
    total = total.view(-1, net.quantization_channels)
    total = net.softmax(total).view(-1)
    _, predict = torch.topk(total.data, 1)
    return predict, state_queue


def generate(model_path,
             model_name,
             generate_path,
             generate_name,
             start_piece=None,
             sr=16000,
             duration=10):
    if os.path.exists(generate_path)is False:
        os.makedirs(generate_path)
    with open('./params/wavenet_params.json', 'r') as f:
        params = json.load(f)
    f.close()
    net = wavenet(**params)
    net = load_model(net, model_path, model_name)
    if start_piece is None:
        start_piece = torch.zeros(1, 256, net.receptive_field)
        start_piece[:, 128, :] = 1.0
        start_piece = Variable(start_piece)
    note_num = duration * sr
    note = start_piece
    state_queue = None
    generated_piece = []
    for i in range(note_num):
        note, state_queue = predict_next(net, note, state_queue)
        note = note[0]
        generated_piece.append(note)
        temp = torch.zeros(1, net.quantization_channels, 1)
        temp[:, note, :] = 1.0
        note = Variable(temp)
    print(generated_piece)
    generated_piece = torch.LongTensor(generated_piece)
    generated_piece = mu_law_decode(generated_piece,
                                    net.quantization_channels)
    generated_piece = generated_piece.numpy()
    wav_name = generate_path + generate_name
    librosa.output.write_wav(wav_name, generated_piece, sr=sr)


generate('./restore/',
         'wavenet14.model',
         './gen/',
         'test.wav',
         duration=10)
