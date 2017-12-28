from model import Discriminator, Generator
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from train import pretrain_generator, pretrain_discriminator, adversarial_train
from utils import recurrent_func, loss_func, get_rewards
import json
import numpy as np
import target
import torch
import torch.optim as optim
import torch.nn as nn


class Fake_Dataset(Dataset):

    def __init__(self):
        self.data = np.random.randint(5000, size=(128, 20))

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()

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
    model_dict = {"generator":generator, "discriminator":discriminator}
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

    return {"worker":w_optimizer, "manager":m_optimizer,
    "discriminator":d_optimizer}

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
    return {"worker":w_scheduler, "manager":m_scheduler,
    "discriminator":d_scheduler}

def prepare_fake_data():
    dataset = Fake_Dataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=4)
    return dataloader

def main(type="discriminator", use_cuda=False):
    if type == "discriminator":
        test_discriminator(use_cuda)
    elif type == "generator":
        test_generator(use_cuda)
    elif type == "loss_func":
        test_loss_func(use_cuda)
    elif type == "target":
        test_target(use_cuda)
    elif type == "dataloader":
        test_dataloader()
    elif type == "train":
        test_train(use_cuda)
    else:
        raise ("Invalid test type!")

def test_discriminator(use_cuda):
    net = Discriminator(
        20, 2, 2000, 64,
        [1, 2, 4, 6, 8, 10, 20],
        [100, 100, 100, 100, 100, 160, 160],
        0, 820, 4, 0.75, 0.2
    )
    print(net)
    sentence = np.random.randint(2000, size=(64, 20))
    sentence = Variable(torch.from_numpy(sentence).long())
    target = np.random.randint(2, size=(64))
    target = Variable(torch.from_numpy(target).long())
    if use_cuda:
        net = net.cuda()
        target = target.cuda()
        sentence = sentence.cuda()
    out_dict = net(sentence)
    print("Disciminator forward test passed.")
    loss_function = nn.CrossEntropyLoss().cuda()
    loss = loss_function(out_dict["score"], target) + net.l2_loss()
    loss.backward()
    print("Disciminator backward test passed.")

def test_generator(use_cuda=False):
    '''
    Prepare model_dict.
    '''
    model_dict = prepare_model_dict(use_cuda)
    '''
    Prepare some fake data.
    '''
    dataloader = prepare_fake_data()

    '''
    Start testing all recurrent functions.
    '''
    for i, sample in enumerate(dataloader):
        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda(async=True)

        # Test pre.
        pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
        for key in pre_rets.keys():
            print("{}:{}".format(key, pre_rets[key].size()))
        print("Pretrain recurrent function test finished!")
        print("\n")
        del pre_rets

        # Test adv.
        adv_rets = recurrent_func('adv')(model_dict, use_cuda)
        for key in adv_rets.keys():
            print("{}:{}".format(key, adv_rets[key].size()))
        print("Adversarial recurrent function test finished!")
        print("\n")
        del adv_rets

        # Test roll.
        gen_token = recurrent_func("rollout")(model_dict, sample,
                                              4, use_cuda)
        print("gen_token:{}".format(gen_token.size()))
        print("Rollout test finished!")
        print("\n")
        del gen_token

        # Test gen.
        gen_token = recurrent_func("gen")(model_dict, use_cuda)
        print("gen_token:{}".format(gen_token.size()))
        print("Generate test finished!")
        print("\n")
        del gen_token

        break

def test_loss_func(use_cuda=False):
    '''
    Prepare model_dict.
    '''
    model_dict = prepare_model_dict(use_cuda)
    generator = model_dict["generator"]
    worker = generator.worker
    manager = generator.manager

    '''
    Prepare some fake data.
    '''
    dataloader = prepare_fake_data()

    '''
    Start testing all recurrent functions.
    '''

    m_optimizer = optim.Adam(manager.parameters(), lr=0.001)
    w_optimizer = optim.Adam(worker.parameters(), lr=0.001)

    m_optimizer.zero_grad()
    w_optimizer.zero_grad()
    for i, sample in enumerate(dataloader):
        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda(async=True)

        # Test pre.
        pre_rets = recurrent_func("pre")(model_dict, sample, use_cuda)
        real_goal = pre_rets["real_goal"]
        prediction = pre_rets["prediction"]
        delta_feature = pre_rets["delta_feature"]

        m_loss = loss_func("pre_manager")(real_goal, delta_feature)
        torch.autograd.grad(m_loss, manager.parameters())
        nn.utils.clip_grad_norm(manager.parameters(), max_norm=5.0)
        m_optimizer.step()
        m_optimizer.zero_grad()

        w_loss = loss_func("pre_worker")(sample, prediction, 5000,
                                         use_cuda)
        torch.autograd.grad(w_loss, worker.parameters())
        nn.utils.clip_grad_norm(worker.parameters(), max_norm=5.0)
        w_optimizer.step()
        w_optimizer.zero_grad()
        print("pre_m_loss={}, pre_w_loss={}".format(
            m_loss.data[0], w_loss.data[0]
        ))
        print("Pretrain loss function test  finished!")
        print("\n")

        # Test adv.
        adv_rets = recurrent_func('adv')(model_dict, use_cuda)
        real_goal = adv_rets["real_goal"]
        all_goal = adv_rets["all_goal"]
        prediction = adv_rets["prediction"]
        delta_feature = adv_rets["delta_feature"]
        delta_feature_for_worker = adv_rets["delta_feature_for_worker"]
        gen_token = adv_rets["gen_token"]
        rewards = get_rewards(model_dict, gen_token, 4, use_cuda)

        m_loss = loss_func("adv_manager")(
            rewards, real_goal, delta_feature
        )
        w_loss = loss_func("adv_worker")(
            all_goal, delta_feature_for_worker, gen_token, prediction, 5000,
            use_cuda
        )

        m_optimizer = optim.Adam(manager.parameters(), lr=0.001)
        w_optimizer = optim.Adam(worker.parameters(), lr=0.001)

        m_optimizer.zero_grad()
        w_optimizer.zero_grad()

        torch.autograd.grad(m_loss, manager.parameters())
        torch.autograd.grad(w_loss, worker.parameters())
        nn.utils.clip_grad_norm(manager.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm(worker.parameters(), max_norm=5.0)
        m_optimizer.step()
        w_optimizer.step()

        print("adv_m_loss={}, adv_w_loss={}".format(
            m_loss.data[0], w_loss.data[0]
        ))
        print("Adversarial training loss function test finished!")
        print("\n")

        if i > 0:
            break

def test_target(use_cuda=False):
    with open("./params/target_params.json", 'r') as f:
        params = json.load(f)
    f.close()
    net = target.Target(**params)
    if use_cuda:
        net = net.cuda()
    dataloader = prepare_fake_data()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    for i, sample in enumerate(dataloader):
        sample = Variable(sample)
        if use_cuda:
            sample = sample.cuda()
        optimizer.zero_grad()
        loss = target.loss_func(net, sample, use_cuda)
        loss.backward()
        optimizer.step()

        if i > 0:
            break
    gen = target.generate(net, use_cuda)
    print(gen.size())
    print("Target test finished!")
    print("\n")


def test_dataloader():
    pass

def test_train(use_cuda):
    '''
    Prepare all needed models, optimizers, schedulers and dataloader.
    '''
    model_dict = prepare_model_dict(use_cuda)
    lr_dict = {"worker":1e-4, "manager":1e-4, "discriminator":1e-4}
    optimizer_dict = prepare_optimizer_dict(model_dict, lr_dict)
    scheduler_dict = prepare_scheduler_dict(optimizer_dict)
    dataloader = prepare_fake_data()

    '''
    Start test all training functions.
    '''
    model_dict, optimizer_dict, scheduler_dict = \
    pretrain_generator(model_dict, optimizer_dict, scheduler_dict,
        dataloader, vocab_size=5258, use_cuda=use_cuda)
    print("Pretrain generator test finished!")

    with open("./params/dis_data_params.json", 'r') as f:
        params = json.load(f)
    if use_cuda:
        params["pin_memory"] = True
    f.close()
    positive_file = params["positive_filepath"]
    negative_file = params["negative_filepath"]

    model_dict, optimizer_dict, scheduler_dict= \
    pretrain_discriminator(model_dict, optimizer_dict, scheduler_dict, params,
        positive_file, negative_file, num_batches=2, num_epochs=1,
        use_cuda=use_cuda)
    print("Pretrain discriminator test finished!")

    adversarial_train(model_dict, optimizer_dict, scheduler_dict,
        params, 5258, positive_file, negative_file, num_batches=2,
        use_cuda=use_cuda)
    print("Adversarial train test finished!")


main('train', use_cuda=False)
