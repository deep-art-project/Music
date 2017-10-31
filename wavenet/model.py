import torch.nn as nn
import torch.nn.functional as F

class wavenet(nn.Module):
    def __init__(self,
                 filter_width,
                 dilations,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 quantization_channels,
                 use_bias):
        '''
        Arguments:
            filter_width: correspoond to kernel_size in nn.Conv1d
                          default 2
            dilations: a list containing all the dilation parameters
            dilation_channels: channels numbers after performing a
                               dilation convolution
            residual_channels: channels after a causal convolution or
                               a dense convolution
            skip_channels: channels after a skip convolution
            quantization_channels: original encoding length of audio files
            use_bias: bool variable, whether to use bias
        '''
        super(wavenet, self).__init__()
        self.filter_width = filter_width
        self.dilations = dilations
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.quantization_channels = quantization_channels
        self.use_bias = use_bias
        self.receptive_field = self.calc_receptive_field()
        self._init_causal_layer()
        self._init_dliation_layer()
        self._init_post_processing_layer()
        self.softmax = nn.Softmax()

    def calc_receptive_field(self):
        return (self.filter_width - 1) * (sum(self.dilations) + 1) + 1

    def _init_causal_layer(self):
        self.causal_layer = nn.Conv1d(self.quantization_channels,
                                      self.residual_channels,
                                      self.filter_width,
                                      bias = self.use_bias)

    def _init_dliation_layer(self):
        self.dilation_layer_stack = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            current = {}
            current['filter'] = nn.Conv1d(self.residual_channels,
                                          self.dilation_channels,
                                          self.filter_width,
                                          dilation = dilation,
                                          bias = self.use_bias)
            current['gate'] = nn.Conv1d(self.residual_channels,
                                        self.dilation_channels,
                                        self.filter_width,
                                        dilation = dilation,
                                        bias = self.use_bias)
            current['dense'] = nn.Conv1d(self.dilation_channels,
                                         self.residual_channels,
                                         1,
                                         bias = self.use_bias)
            current['skip'] = nn.Conv1d(self.dilation_channels,
                                        self.skip_channels,
                                        1,
                                        bias = self.use_bias)
            self.dilation_layer_stack.extend(list(current.values()))

    def _init_post_processing_layer(self):
        self.post_process_1 = nn.Conv1d(self.skip_channels,
                                      self.skip_channels,
                                      1,
                                      bias = self.use_bias)
        self.post_process_2 = nn.Conv1d(self.skip_channels,
                                        self.quantization_channels,
                                        1,
                                        bias = self.use_bias)

    def forward(self, wave_sample):
        '''
        Argument:
            wave_sample: one_hot encoded Variable
                         the third dimension of wave_sample must
                         be at least self.receptive_field + 1
        Return:
            Assume the third dimension of wave_sample is l(sequence length)
            return a Variable of size (l - self.receptive_field + 1)
            * self.quantization_channels, each line representing the
            softmax probability of the next sound piece
        '''
        batch_size, original_channels, seq_len = wave_sample.size()
        output_width = seq_len - self.receptive_field + 1
        if output_width <= 0:
            raise ValueError("wave sample not long enough")

        #First, pass through a causal convolution layer
        current_out = self.causal_layer(wave_sample)

        #Then pass through all dilation convolution layers
        skip_contribution_stack = []
        for i, dilation in enumerate(self.dilations):
            #calculate the output of a dilation layer
            #with residual connection
            current_in = current_out
            j = 4 * i
            filter_layer, gate_layer, dense_layer, skip_layer = \
                self.dilation_layer_stack[j], \
                self.dilation_layer_stack[j + 1],\
                self.dilation_layer_stack[j + 2],\
                self.dilation_layer_stack[j + 3]
            current_filter = filter_layer(current_in)
            current_gate = gate_layer(current_in)
            combined = F.sigmoid(current_gate) * F.tanh(current_filter)
            current_dense = dense_layer(combined)
            _, _, current_len = current_dense.size()
            current_in_sliced = current_in[:, :, -current_len:]
            current_out = current_dense + current_in_sliced

            #Then calculate the skip contributions to form the prediction
            skip = combined[:, :, -output_width:]
            skip = skip_layer(skip)
            skip_contribution_stack.append(skip)

        #calculate the final prediction
        #first take the sum of all skip contributions
        #then use two relu activations as two 1*1 convolution
        total = sum(skip_contribution_stack)
        total = F.relu(total)
        total = self.post_process_1(total)
        total = F.relu(total)
        total = self.post_process_2(total)

        #Finally, for each row in total, perform a softmax
        #to get probability
        batch_size, channels, seq_len = total.size()
        total = total.view(-1, self.quantization_channels)
        total = self.softmax(total)
        return total

def predict_next(model, wave_var, quantization_channels = 256):
    '''
    Arguments:
        model: a pretrained wavenet model
        wave_var: type(torch.autograd.Variable)
                  size(batch_size * quantization_channels * wave_length)
                  must be one_hot encoding form
    Return:
        A Tensor of size quantization_channels
        Each value in the tensor representing the probability of the
        next piece of sound being the i'th category
    '''
    raw_out = model(wave_var)
    out = raw_out.view(-1, quantization_channels)
    last = out[-1, :]
    last = last.view(-1)
    return last


