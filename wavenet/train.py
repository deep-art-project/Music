from audio_data import audio_data_loader, one_hot_encode
from functools import cmp_to_key
from model import wavenet
from torch.autograd import Variable
import glob
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

def get_params(json_dir):
    with open(json_dir, 'r') as f:
        params = json.load(f)
    f.close()
    return params

def get_arguments():
    train_params = get_params('./params/train_params.json')
    wavenet_params = get_params('./params/wavenet_params.json')
    dataset_params = get_params('./params/dataset_params.json')
    return train_params, wavenet_params, dataset_params

def get_optimizer(model, optimizer_type, learning_rate, momentum):
    if optimizer_type == 'sgd':
        return optim.SGD(model.parameters(),
                         lr = learning_rate,
                         momentum = momentum)
    if optimizer_type == 'rmsprop':
        return optim.RMSprop(model.parameters(),
                            lr = learning_rate,
                            momentum = momentum)
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(),
                          lr = learning_rate)

def save_model(model, num_iter, path):
    model_name = "wavenet" + str(num_iter) + ".model"
    checkpoint_path = path + model_name
    print("Storing checkpoint to {} ...".format(path))
    torch.save(model.state_dict(), checkpoint_path)
    print("Done!")

def load_model(model, path, model_name):
    checkpoint_path = path + model_name
    print("Trying to restore saved checkpoint from ",\
          "{}".format(checkpoint_path))
    model_state_dict = torch.load(checkpoint_path)
    if model_state_dict:
        print("Checkpoint found, restoring!")
        model.load_state_dict(model_state_dict)
        return model
    else:
        print("No checkpoint found!")
        return None

def train():
    '''
    Check whether cuda is available.
    '''
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cudnn.benchmark = True

    '''
    Get all needed parameters.
    All parameters are stored in json file in directory './params'.
    If you want to change the settings, simply modify the json file
    in './params/'
    '''
    train_params, wavenet_params, dataset_params = get_arguments()

    '''
    Launch instances of wavenet model and dataloader.
    '''
    net = wavenet(**wavenet_params)
    epoch_trained = 0
    if train_params["restore_model"]:
        net = load_model(net,
                         train_params["restore_dir"],
                         train_params["restore_model"])
        if net == None:
            print("Initialize network and train from scratch.")
            net = wavenet(**wavenet_params)
        else:
            epoch_trained = train_params["restore_model"].split('.')[0]
            epoch_trained = int(epoch_trained[7:])
    dataloader = audio_data_loader(**dataset_params)

    '''
    Whether use gpu to train the network.
    whether use multi-gpu to train the network.
    '''
    if cuda_available == False and train_params["device_ids"] != None:
        raise ValueError("Cuda is not avalable,"+\
                         " can not train model using multi-gpu.")
    if cuda_available:
        if train_params["device_ids"]:
            batch_size = dataset_params["batch_size"]
            num_gpu = len(train_params["device_ids"])
            assert batch_size % num_gpu == 0
            net = nn.DataParallel(net,device_ids = train_params["device_ids"])
        net = net.cuda()

    '''
    Start training.
    Save the model per train_params["check_point_every"] epochs.
    Save model to train_params["restore_dir"].
    Save at most train_params["max_check_points"] models.
    If the number of models in restore_dir is over max_check_points,
    overwrite the oldest model with the newest one.
    Write logging information to train_params["log_dir"].
    Logging information includes one epoch's average loss
    '''
    print("Start training.")
    print("Writing logging information to ",\
          "{}".format(train_params["log_dir"]))
    print("Models are saved in {}".format(train_params["restore_dir"]))

    '''
    Define optimizer and loss function.
    '''
    optimizer = get_optimizer(net,
                              train_params["optimizer"],
                              train_params["learning_rate"],
                              train_params["momentum"])
    loss_func = nn.CrossEntropyLoss()
    if cuda_available:
        loss_func = loss_func.cuda()
    if not os.path.exists(train_params["log_dir"]):
        os.makedirs(train_params["log_dir"])
    if not os.path.exists(train_params["restore_dir"]):
        os.makedirs(train_params["restore_dir"])
    loss_log_file = open(train_params["log_dir"]+'loss_log.log', 'w')
    store_log_file = open(train_params["log_dir"]+'store_log.log', 'w')

    '''
    Train in epochs
    '''
    for epoch in range(train_params["num_epochs"]):
        total_loss = 0.0
        for i_batch, sampled_batch in enumerate(dataloader):
            for j in range(len(sampled_batch)):
                optimizer.zero_grad()
                piece, target = one_hot_encode(sampled_batch[j],
                                               cuda_available)
                piece, target = Variable(piece), Variable(target)
                logits = net(piece)
                loss = loss_func(logits, target)
                total_loss += loss.data[0]
                loss.backward()
                optimizer.step()
            print("Finish one song!")
        avg_loss = total_loss / i_batch
        line = "Average loss of epoch " + str(epoch_trained + epoch + 1) +\
               " is " + str(avg_loss) + "\n"
        loss_log_file.writelines(line)
        loss_log_file.flush()

        '''
        Store model per check_point_every epochs.
        '''
        if (epoch + 1) % train_params["check_point_every"] == 0:
            stored_models = glob.glob(train_params["restore_dir"] +\
                                      "*.model")
            #First whether to delete one oldest model
            if len(stored_models) == train_params["max_check_points"]:
                def cmp(x, y):
                    x = x.split('/')[-1]
                    y = y.split('/')[-1]
                    x = x.split('.')[0]
                    y = y.split('.')[0]
                    x = int(x[7:])
                    y = int(y[7:])
                    return x - y
                stored_models = sorted(stored_models,
                                       key = cmp_to_key(cmp))
                os.remove(stored_models[0])
            #Then store the newest model
            save_model(net, epoch_trained + epoch + 1,
                       train_params["restore_dir"])
            line = "Epoch " + str(epoch_trained + epoch + 1) + \
                   ", model saved!\n"
            store_log_file.writelines(line)
            store_log_file.flush()
    loss_log_file.close()
    store_log_file.close()

if __name__ == '__main__':
    train()
