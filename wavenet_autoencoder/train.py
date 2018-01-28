import warnings
import torch 
from collections import OrderedDict
import json
import os
import glob
from faster_audio_data import audio_data_loader
from functools import cmp_to_key
from model1 import wavenet_autoencoder
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

def get_params(json_dir):
	with open(json_dir,'r') as f:
		params = json.load(f)
	f.close()
	return params

def get_arguments():
	train_params = get_params('./params/train_params.json')
	model_params=get_params('./params/model_params.json')
	dataset_params = get_params('./params/dataset_params.json')
	return train_params,model_params,dataset_params

def get_optimizer(model,optimizer_type,learning_rate,momentum1=False):
	if optimizer_type =='sgd':
		return optim.sgd(model.parameters(),lr=learning_rate,momentum=momentum1)
	if optimizer_type == 'RMSprop':
		return optim.RMSprop(model.parameters(),lr=learning_rate,momentum= momentum1)
	if optimizer_type == 'Adam':
		return optim.Adam(model.parameters(),lr=learning_rate)
	if optimizer_type == 'lbfgs':
                return optim.LBFGS(model.parameters(), lr = learning_rate)	

def save_model(model,num_epoch,path):
	model_name = 'wavenet_autoencoder' + str(num_epoch)+'.model'
	checkpoint_path = path + model_name
	print('Storing checkpoint to {}...'.format(path))
	torch.save(model.state_dict(),checkpoint_path)
	print('done')


def load_model(model, path, model_name):
    checkpoint_path = path + model_name
    print("Trying to restore saved checkpoint from ",
          "{}".format(checkpoint_path))
    if os.path.exists(checkpoint_path):
        print("Checkpoint found, restoring!")
        # Create a new state dict to prevent error when storing a model
        # on one device and restore it from another
        state_dict = torch.load(checkpoint_path)
        keys = list(state_dict.keys())
        if keys[0][:6] == 'module':
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        return model
    else:
        print("No checkpoint found!")
        return None

def train():
	cuda_available = torch.cuda.is_available()
	train_params,model_params,dataset_params = get_arguments()
	net = wavenet_autoencoder(**model_params)
	epoch_trained = 0
	if train_params['restore_model']:
		net = load_model(net,train_params['restore_dir'],train_params['restore_model'])
		if net is None:
			print("Initialize network and train from scratch.")
			net = wavenet_autoencoder(**model_params)
		else:
			epoch_trained = train_params["restore_model"].split('.')[0]
			epoch_trained = int(epoch_trained[7:])
	dataloader = audio_data_loader(**dataset_params)

	if cuda_available is False :
		warnings.warn("Cuda is not avalable, can not train model using multi-gpu.")
	if cuda_available:
		if train_params["device_ids"]:
			batch_size = dataset_params["batch_size"]
			num_gpu = len(train_params["device_ids"])
			assert batch_size % num_gpu == 0
			net = nn.DataParallel(net,device_ids=train_params['device_ids'])
		torch.backends.cudnn.benchmark = True		
		net = net.cuda()



	optimizer = get_optimizer(net,train_params['optimizer_type'],train_params['learning_rate'],train_params['momentum'])

	loss_func = nn.CrossEntropyLoss()
	if cuda_available:
		loss_func=loss_func.cuda()
	if not os.path.exists(train_params['log_dir']) :
		os.makedirs(train_params['log_dir'])
	if not os.path.exists(train_params['restore_dir']):
		os.makedirs(train_params['restore_dir'])
	loss_log_file = open(train_params['log_dir']+'loss_log.log','a')
	store_log_file = open(train_params['log_dir']+'store_log.log','a')


	total_loss = 0
	with open(train_params['log_dir']+'loss_log.log','r') as f:
		lines = f.readlines()
		if len(lines) > 0:
			num_trained = lines[-1].split(' ')[2]
			num_trained = int(num_trained)
		else:
			num_trained = 0
	f.close()

	for epoch in range(train_params['num_epochs']):
		for i_batch,sample_batch in enumerate(dataloader):
		#	print(sample_batch)
			optimizer.zero_grad()
			music_piece = sample_batch['audio_piece']
			target_piece = sample_batch['audio_target']
			if cuda_available:
				music_piece = music_piece.cuda(async=True)
				target_piece = target_piece.cuda(async=True)
			music_piece = Variable(music_piece)
			target_piece = Variable(target_piece.view(-1))
		#	print(music_piece.size())
	#		print('it is ok1')
			outputs = net(music_piece)
		#	print(outputs.size())
	#		print('it is ok')
			loss = loss_func(outputs,target_piece)
			total_loss += loss.data[0]
			loss.backward()
			optimizer.step()

			num_trained += 1

			if num_trained%train_params['print_every'] ==0:
				avg_loss = total_loss/train_params['print_every']
				line = 'Average loss is ' + str(avg_loss) +'\n'
				loss_log_file.writelines(line)
				loss_log_file.flush()
				total_loss =0

		if (epoch+1)%train_params['check_point_every'] ==0:
			stored_models = glob.glob(train_params['restore_dir']+'*.model')
			if len(stored_models) == train_params['max_check_points']:
				def cmp(x,y):
					x=os.path.splitext(x)[0]
					x=os.path.split(x)[-1]
					y=os.path.splitext(y)[0]
					y=os.path.split(y)[-1]
					x=int(x[7:])
					y=int(y[7:])
					return x-y

				sorted_models = sorted(stored_models,keys=cmp_to_key(cmp))
				os.remove(sorted_models[0])
			print(epoch_trained)
			save_model(net,epoch_trained + epoch + 1,train_params['restore_dir'])
			line = 'Epoch' + str(epoch_trained+epoch+1) +'model saved!'
			store_log_file.writelines(line)
			store_log_file.flush()
	loss_log_file.close()
	store_log_file.close()

    	
if __name__ == '__main__':
    train()



























