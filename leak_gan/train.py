from data import real_data_loader, dis_data_loader
from model import Discriminator, Generator
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from utils import recurrent_func, loss_func, get_sample, get_rewards
import glob
import json
import numpy as np
import os
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
    dis_data_params = get_params("./params/dis_data_params.json")
    real_data_params = get_params("./params/real_data_params.json")
    return {
        "train_params": train_params,
        "leak_gan_params": leak_gan_params,
        "target_params": target_params,
        "dis_data_params": dis_data_params,
        "real_data_params": real_data_params
    }


def prepare_model_dict(use_cuda=False):
    f = open("./params/leak_gan_params.json")
    params = json.load(f)
    f.close()
    discriminator_params = params["discriminator_params"]
    generator_params = params["generator_params"]
    worker_params = generator_params["worker_params"]
    manager_params = generator_params["manager_params"]
    discriminator_params["goal_out_size"] = sum(
        discriminator_params["num_filters"]
    )
    worker_params["goal_out_size"] = discriminator_params["goal_out_size"]
    manager_params["goal_out_size"] = discriminator_params["goal_out_size"]
    discriminator = Discriminator(**discriminator_params)
    generator = Generator(worker_params, manager_params,
                          generator_params["step_size"])
    if use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    model_dict = {"generator": generator, "discriminator": discriminator}
    return model_dict


def prepare_optimizer_dict(model_dict, lr_dict):
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    worker = generator.worker
    manager = generator.manager

    m_lr = lr_dict["manager"]
    w_lr = lr_dict["worker"]
    d_lr = lr_dict["discriminator"]

    w_optimizer = optim.Adam(worker.parameters(), lr=w_lr)
    m_optimizer = optim.Adam(manager.parameters(), lr=m_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    return {"worker": w_optimizer, "manager": m_optimizer,
            "discriminator": d_optimizer}


def prepare_scheduler_dict(optmizer_dict, step_size=200, gamma=0.99):
    w_optimizer = optmizer_dict["worker"]
    m_optimizer = optmizer_dict["manager"]
    d_optimizer = optmizer_dict["discriminator"]

    w_scheduler = optim.lr_scheduler.StepLR(w_optimizer, step_size=step_size,
                                            gamma=gamma)
    m_scheduler = optim.lr_scheduler.StepLR(m_optimizer, step_size=step_size,
                                            gamma=gamma)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=step_size,
                                            gamma=gamma)
    return {"worker": w_scheduler, "manager": m_scheduler,
            "discriminator": d_scheduler}


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
                loss = cross_entropy(outs["score"], label.view(-1)) + \
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


def save_checkpoint(model_dict, optimizer_dict, scheduler_dict,
                    ckpt_num, replace=False):
    filename = "cpkt" + str(ckpt_num) + ".pth.tar"
    torch.save({"model_dict": model_dict, "optimizer_dict": optimizer_dict,
        "scheduler_dict": scheduler_dict, "ckpt_num": ckpt_num}, filename)
    if replace:
        ckpts = glob.glob("cpkt*")
        ckpt_nums = [int(x.split('.')[0][4:]) for x in ckpts]
        oldest_ckpt = "ckpt" + str(min(ckpt_nums)) + ".pth.tar"
        os.remove(oldest_ckpt)


def restore_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    return ckpt


def main():
    '''
    Get all parameters.
    '''
    param_dict = get_arguments()
    use_cuda = torch.cuda.is_available()

    '''
    Set random seed.
    '''
    torch.manual_seed(param_dict["train_params"]["seed"])

    '''
    Pretrain step
    '''
    ckpt_path = param_dict["train_params"]["ckpt_path"]
    if ckpt_path is not None:
        ckpt = restore_checkpoint(ckpt_path)
        model_dict = ckpt["model_dict"]
        optimizer_dict = ckpt["optimizer_dict"]
        scheduler_dict = ckpt["scheduler_dict"]
        ckpt_num = ckpt["ckpt_num"]

    else:
        model_dict = prepare_optimizer_dict(use_cuda)
        lr_dict = param_dict["train_params"]["lr_dict"]
        optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)
        gamma = param_dict["train_params"]["decay_rate"]
        step_size = param_dict["train_params"]["decay_step_size"]
        scheduler_dict = prepare_scheduler_dict(optimizer_dict,
                                                gamma=gamma,
                                                step_size=step_size)
        ckpt_num = 0

    '''
    Pretrain discriminator.
    '''
    with open("./params/dis_data_params.json", 'r') as f:
        dis_data_params = json.load(f)
    if use_cuda:
        dis_data_params["pin_memory"] = True
    f.close()
    positive_file = dis_data_params["positive_filepath"]
    negative_file = dis_data_params["negative_filepath"]
    num_batches = param_dict["train_params"]["generate_num"]

    for _ in range(param_dict["train_params"]["pre_dis_epoch_num"]):
        model_dict, optimizer_dict, scheduler_dict = \
        pretrain_discriminator(
            model_dict, optimizer_dict, scheduler_dict, dis_data_params,
            positive_file, negative_file, num_batches=num_batches,
            num_epochs=1, use_cuda=use_cuda)
    '''
    Pretrain generator
    '''
    real_data_params = param_dict["real_data_params"]
    vocab_size = param_dict["leak_gan_params"]["discriminator"]["vocab_size"]
    if use_cuda:
        real_data_params["pin_memory"] = True
    r_dataloader = real_data_loader(**real_data_params)
    for _ in range(param_dict["train_params"]["pre_gen_epoch_num"]):
        model_dict, optimizer_dict, scheduler_dict = \
        pretrain_generator(model_dict, optimizer_dict, scheduler_dict,
            r_dataloader, vocab_size=vocab_size, use_cuda=use_cuda)
    '''
    Finish pretrain, save checkpoint
    '''
    cpkt_num = 0
    save_checkpoint(model_dict, optimizer_dict, scheduler_dict, ckpt_num)

    '''
    Start adversarial train_params
    '''
    save_num = param_dict["train_params"]["save_num"]
    replace_num = param_dict["train_params"]["replace_num"]
    for epoch in range(param_dict["train_params"]["total_epoch"]):
        model_dict, optimizer_dict, scheduler_dict = \
        adversarial_train(model_dict, optimizer_dict, scheduler_dict,
            dis_data_params, vocab_size, positive_file, negative_file,
            num_batches, use_cuda=use_cuda)
        if (epoch + 1) % save_num == 0:
            ckpt_num += 1
            if ckpt_num % replace_num == 0:
                save_checkpoint(model_dict, optimizer_dict, scheduler_dict,
                            ckpt_num, replace=True)
            else:
                save_checkpoint(model_dict, optimizer_dict, scheduler_dict,
                            ckpt_num)


main()
