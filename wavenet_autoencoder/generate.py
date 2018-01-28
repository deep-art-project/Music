from audio_func import mu_law_decode
from collections import OrderedDict
from train import load_model
import json
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F 
import torch
import os
from model1 import wavenet_autoencoder


def predict_next(net,input_wav,quantization_channel = 256):
	out_wav = net(input_wav)
	out = out_wav.view(-1,quantization_channel)
	last = out[-1,:]
	last = last.view(-1)
	_,predict = torch.topk(last, 1)
	return int(predict)


def generate(model_path,model_name,generate_path,generate_name,start_piece=None,sr = 16000,duration=10):
	if os.path.exists(generate_path) is False:
		os.makedirs(generate_path)
	with open('./params/model_params.json') as f :
		model_params = json.load(f)
	f.close()
	net = wavenet_autoencoder(**model_params)
	net = load_model(net,model_path,model_name)
	cuda_available = torch.cuda.is_available()
	if cuda_available is True:
		net = net.cuda()
#	print(net.receptive_field)
	if start_piece is None:
		start_piece = torch.zeros(1, 256, net.receptive_field+512)
		start_piece[:, 128, :] = 1.0
		start_piece = Variable(start_piece)
	if cuda_available is True:
		start_piece = start_piece.cuda()
	note_num = duration * sr
	note = start_piece
	state_queue = None
	generated_piece = []
	input_wav = start_piece
	for i in range(note_num):
		print(i)
		predict_note = predict_next(net, input_wav)
		generated_piece.append(note)
		temp = torch.zeros(net.quantization_channel,1)
		temp[predict_note] =1
		temp = temp.view(1,net.quantization_channel,1)
#		temp = torch.zeros(1, net.quantization_channel, 1)
#		temp[:, predict_note, :] = 1.0
		note = Variable(temp)
		note = note.cuda()
#		print(note.size())
#		print(input_wav.size())
		input_wav = torch.cat((input_wav[:,-net.receptive_field-511:],note), 2)
	print(generated_piece)
	generated_piece = torch.LongTensor(generated_piece)
	generated_piece = mu_law_decode(generated_piece,
							net.quantization_channel)
	generated_piece = generated_piece.numpy()
	wav_name = generate_path + generate_name
	librosa.output.write_wav(wav_name, generated_piece, sr=sr)

if __name__ =='__main__':
	generate('./restore/', 'wavenet_autoencoder1.model','./generate/','generate1' )
