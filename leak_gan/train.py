from data import real_data_loader, dis_data_loader
from model import Discriminator, Generator
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from utils import recurrent_func, loss_func, get_sample, get_rewards
import json
import numpy as np
import target
import torch
import torch.nn as nn
import torch.optim as optim


def get_params(filepath):
    with open(filepath, 'r') as f:
        params = json.load(f)
    f.close()
    return params

def get_arguments():
    train_params = get_params("./params/train_params.json")
    leak_gan_params = get_params("./params/leak_gan_params.json")
    target_params = get_params("./params/target_params.json")
    return {
        "train_params":train_params,
        "leak_gan_params":leak_gan_params,
        "target_params":target_params
    }

def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)

def pretrain_generator(model_dict, optimizer_dict, scheduler_dict,
                       dataloader, vocab_size, max_norm=5.0, use_cuda=False):
    '''
    Get models, optimizers and schedulers.
    '''
    generator = model_dict["generator"]
    worker = generator.worker
    manager = generator.manager

    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]

    m_optimizer.zero_grad()
    w_optimizer.zero_grad()

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]

    '''
    Perform pretrain step for real data.
    '''
    for i, sample in enumerate(dataloader):
        m_lr_scheduler.step()
        w_lr_scheduler.step()

        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda(async=True)

        # Calculate pretrain loss.
        pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
        real_goal = pre_rets["real_goal"]
        prediction = pre_rets["prediction"]
        delta_feature = pre_rets["delta_feature"]

        m_loss = loss_func("pre_manager")(real_goal, delta_feature)
        torch.autograd.grad(m_loss, manager.parameters())
        clip_grad_norm(manager.parameters(), max_norm=max_norm)
        m_optimizer.step()
        m_optimizer.zero_grad()

        w_loss = loss_func("pre_worker")(sample, prediction, vocab_size,
                                         use_cuda)
        torch.autograd.grad(w_loss, worker.parameters())
        clip_grad_norm(worker.parameters(), max_norm=max_norm)
        w_optimizer.step()
        w_optimizer.zero_grad()

    '''
    Update model_dict, optimizer_dict and scheduler_dict.
    '''
    generator.worker = worker
    generator.manager = manager
    model_dict["generator"] = generator

    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer

    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler

    return model_dict, optimizer_dict, scheduler_dict

def generate_samples(model_dict, negative_file, num_batches,
                     use_cuda=False, temperature=1.0):
    neg_data = []
    for i in range(num_batches):
        sample = get_sample(model_dict, use_cuda, temperature)
        sample = sample.cpu()
        neg_data.append(sample.data.numpy())
    neg_data = np.concatenate(neg_data, axis=0)
    np.save(negative_file, neg_data)

def pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict,
                           dis_dataloader_params, positive_file,
                           negative_file, num_batches, num_epochs,
                           use_cuda=False, temperature=1.0):
    discriminator = model_dict["discriminator"]

    d_optimizer = optimizer_dict["discriminator"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    generate_samples(model_dict, negative_file, num_batches, use_cuda,
                     temperature)
    dis_dataloader_params["positive_filepath"] = positive_file
    dis_dataloader_params["negative_filepath"] = negative_file
    dataloader = dis_data_loader(**dis_dataloader_params)

    cross_entropy = nn.CrossEntropyLoss()
    if use_cuda:
        cross_entropy = cross_entropy.cuda()

    for epoch in range(num_epochs):
        for i, sample in enumerate(dataloader):
            d_optimizer.zero_grad()
            data, label = sample["data"], sample["label"]
            data = Variable(data)
            label = Variable(label)
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            outs = discriminator(data)

            loss = cross_entropy(outs["score"], label.view(-1)) + \
                   discriminator.l2_loss()
            d_lr_scheduler.step()
            loss.backward()
            d_optimizer.step()

    model_dict["discriminator"] = discriminator
    optimizer_dict["discriminator"] = d_optimizer
    scheduler_dict["discriminator"] = d_lr_scheduler
    return model_dict, optimizer_dict, scheduler_dict

def adversarial_train(model_dict, optimizer_dict, scheduler_dict,
                      dis_dataloader_params, vocab_size, positive_file,
                      negative_file, num_batches, gen_train_num=1,
                      dis_train_epoch=5, dis_train_num=3, max_norm=5.0,
                      rollout_num=4, use_cuda=False, temperature=1.0):
    '''
    Get models, optimizers and schedulers.
    '''
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_optimizer = optimizer_dict["manager"]
    w_optimizer = optimizer_dict["worker"]
    d_optimizer = optimizer_dict["discriminator"]

    m_optimizer.zero_grad()
    w_optimizer.zero_grad()

    m_lr_scheduler = scheduler_dict["manager"]
    w_lr_scheduler = scheduler_dict["worker"]
    d_lr_scheduler = scheduler_dict["discriminator"]

    '''
    Adversarial train for generator.
    '''
    for _ in range(gen_train_num):
        m_lr_scheduler.step()
        w_lr_scheduler.step()

        m_optimizer.zero_grad()
        w_optimizer.zero_grad()

        adv_rets = recurrent_func('adv')(model_dict, use_cuda)
        real_goal = adv_rets["real_goal"]
        all_goal = adv_rets["all_goal"]
        prediction = adv_rets["prediction"]
        delta_feature = adv_rets["delta_feature"]
        delta_feature_for_worker = adv_rets["delta_feature_for_worker"]
        gen_token = adv_rets["gen_token"]
        rewards = get_rewards(model_dict, gen_token, rollout_num, use_cuda)

        m_loss = loss_func("adv_manager")(
            rewards, real_goal, delta_feature
        )
        w_loss = loss_func("adv_worker")(
            all_goal, delta_feature_for_worker, gen_token, prediction,
            vocab_size, use_cuda
        )

        torch.autograd.grad(m_loss, manager.parameters())
        torch.autograd.grad(w_loss, worker.parameters())
        clip_grad_norm(manager.parameters(), max_norm=max_norm)
        clip_grad_norm(worker.parameters(), max_norm=max_norm)
        m_optimizer.step()
        w_optimizer.step()

    del adv_rets
    del real_goal
    del all_goal
    del prediction
    del delta_feature
    del delta_feature_for_worker
    del gen_token
    del rewards

    '''
    Adversarial train for discriminator.
    '''
    for _ in range(dis_train_epoch):
        generate_samples(model_dict, negative_file, num_batches,
                         use_cuda, temperature)
        dis_dataloader_params["positive_filepath"] = positive_file
        dis_dataloader_params["negative_filepath"] = negative_file
        dataloader = dis_data_loader(**dis_dataloader_params)

        cross_entropy = nn.CrossEntropyLoss()
        if use_cuda:
            cross_entropy = cross_entropy.cuda()

        for _ in range(dis_train_num):
            for i, sample in enumerate(dataloader):
                data, label = sample["data"], sample["label"]
                data = Variable(data)
                label = Variable(label)
                if use_cuda:
                    data = data.cuda(async=True)
                    label = label.cuda(async=True)
                outs = discriminator(data)
                loss = cross_entropy(outs["score"], label) + \
                       discriminator.l2_loss()
                d_optimizer.zero_grad()
                d_lr_scheduler.step()
                loss.backward()
                d_optimizer.step()

    model_dict["discriminator"] = discriminator
    generator.worker = worker
    generator.manager = manager
    model_dict["generator"] = generator

    optimizer_dict["manager"] = m_optimizer
    optimizer_dict["worker"] = w_optimizer
    optimizer_dict["discriminator"] = d_optimizer

    scheduler_dict["manager"] = m_lr_scheduler
    scheduler_dict["worker"] = w_lr_scheduler
    scheduler_dict["discriminator"] = d_lr_scheduler

    return model_dict, optimizer_dict, scheduler_dict