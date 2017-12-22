from scipy.special import expit
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_vars(generator, discriminator, use_cuda=False):
    h_w_t, c_w_t = generator.init_hidden()
    h_m_t, c_m_t = generator.init_hidden()
    last_goal = Variable(torch.zeros(
        generator.worker.batch_size, generator.worker.goal_out_size
    ))
    real_goal = generator.manager.goal_init
    x_t = Variable(nn.init.constant(torch.Tensor(
        generator.worker.batch_size
    ), discriminator.start_token)).long()
    vs = [h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t]
    if use_cuda:
        for var in vs:
            var = var.cuda(async=True)
    return vs

def recurrent_func(f_type='pre'):
    '''
    There are three types of recurrent function:
        'pre' ------ pretrain
        'adv' ------ adversarial train
        'rollout' ------ rollout for evaluate reward

    Return a function of the corresponding type
    '''
    if f_type == 'pre':
        def func(model_dict, real_data, use_cuda=False, temperature=1.0):
            '''
            Get generator and discriminator
            '''
            generator = model_dict['generator']
            discriminator = model_dict['discriminator']

            '''
            Initialize variables and lists for forward step.
            '''
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            feature_list = []
            delta_feature_list = []
            prediction_list = []
            real_goal_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size

            '''
            Perform forward step for pretraining generator and
            discriminator.
            '''
            while t < seq_len + 1 :
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size)
                    ).long()
                else:
                    cur_sen = real_data[:, :t]
                    cur_sen = cur_sen.contiguous()
                    cur_sen = F.pad(
                        cur_sen.view(-1, t), (0, seq_len - t), value=vocab_size
                    )
                if use_cuda:
                    cur_sen = cur_sen.cuda(async=True)
                f_t = discriminator(cur_sen)["feature"]

                '''
                Generator forward step.
                '''
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator(
                        x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, 1.0
                    )
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(
                        batch_size, goal_out_size
                    ))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                    real_goal_list.append(real_goal)

                '''
                Store needed information for caculating loss function
                '''
                feature_list.append(f_t)
                prediction_list.append(probs)
                if t > 0:
                    if t % step_size == 0:
                        delta_feature_list.append(
                            f_t - feature_list[t - step_size]
                        )
                t = t_

            '''
            Post process and return variables needed for calculating loss.
            '''
            if len(real_goal_list) == len(delta_feature_list) + 1:
                real_goal_list = real_goal_list[:-1]
            prediction_list = prediction_list[:-1]
            real_goal_var = torch.stack(real_goal_list).permute(1, 0, 2)
            prediction_var = torch.stack(prediction_list).permute(1, 0, 2)
            delta_feature_var = torch.stack(delta_feature_list).permute(1, 0, 2)
            rets = {"real_goal":real_goal_var,
                    "prediction":prediction_var,
                    "delta_feature":delta_feature_var}
            for ret in rets.values():
                if ret.is_contiguous():
                    ret = ret.contiguous()
            return rets
        return func

    elif f_type == 'adv':
        def func(model_dict, use_cuda=False, temperature=1.0):
            '''
            Get generator and discriminator
            '''
            generator = model_dict['generator']
            discriminator = model_dict['discriminator']

            '''
            Initialize variables and lists for forward step.
            '''
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            feature_list = []
            delta_feature_list = [] # f_(t+c) - f_t
            delta_feature_for_worker_list = [] # f_t - f_(t-i)
            prediction_list = []
            real_goal_list = []
            all_goal_list = []
            gen_token_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size

            '''
            Perform forward step for adversarial training for dicriminator and
            generator.
            '''
            while t < seq_len + 1:
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t), value=vocab_size
                    )
                f_t = discriminator(cur_sen)["feature"]
                '''
                Generator forward step.
                '''
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator(
                        x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, temperature
                    )
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(
                        batch_size, goal_out_size
                    ))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                    real_goal_list.append(real_goal)

                '''
                Store needed information for calculating loss function
                '''
                feature_list.append(f_t)
                prediction_list.append(probs)
                if t > 0:
                    if t % step_size == 0:
                        delta_feature_list.append(
                            f_t - feature_list[t - step_size]
                        )
                        delta_feature_for_worker_list.append(
                            f_t - feature_list[t - step_size]
                        )
                    else:
                        delta_feature_for_worker_list.append(
                            f_t - feature_list[t - t % step_size]
                        )
                    all_goal_list.append(real_goal)
                gen_token_list.append(x_t)
                t = t_

            '''
            Post process and return variables.
            '''
            if len(real_goal_list) == len(delta_feature_list) + 1:
                real_goal_list = real_goal_list[:-1]
            prediction_list = prediction_list[:-1]
            gen_token_list = gen_token_list[:-1]
            real_goal_var = torch.stack(real_goal_list).permute(1, 0, 2)
            all_goal_var = torch.stack(all_goal_list).permute(1, 0, 2)
            prediction_var = torch.stack(prediction_list).permute(1, 0, 2)
            delta_feature_var = torch.stack(
                delta_feature_list
            ).permute(1, 0, 2)
            gen_token_var = torch.stack(gen_token_list).permute(1, 0)
            delta_feature_for_worker_var = torch.stack(
                delta_feature_for_worker_list
            ).permute(1, 0, 2)
            rets = {"real_goal":real_goal_var,
                    "all_goal":all_goal_var,
                    "prediction":prediction_var,
                    "delta_feature":delta_feature_var,
                    "delta_feature_for_worker":delta_feature_for_worker_var,
                    "gen_token":gen_token_var}
            for ret in rets.values():
                if ret.is_contiguous():
                    ret = ret.contiguous()
            return rets
        return func

    elif f_type == 'rollout':
        def func(model_dict, input_x, given_num, use_cuda=False,
                 temperature=1.0):
            '''
            Get generator and discriminator
            '''
            generator = model_dict['generator']
            discriminator = model_dict['discriminator']

            '''
            Initialize variables and lists for forward step.
            '''
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            gen_token_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size

            '''
            Use input_x to perform generator forward step.
            '''
            while t < given_num + 1:
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t), value=vocab_size
                    )
                f_t = discriminator(cur_sen)["feature"]

                '''
                Generator forward step.
                '''
                _, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator(
                        x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, temperature
                    )
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(
                        batch_size, goal_out_size
                    ))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                if t < given_num:
                    x_t = input_x[:, t].contiguous()
                    gen_token_list.append(x_t)
                t = t_

            '''
            Perform rollout.
            '''
            while t < seq_len + 1:

                '''
                Extract feature f_t.
                '''
                if len(gen_token_list) == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t + 1), value=vocab_size
                    )
                f_t = discriminator(cur_sen)["feature"]

                '''
                Generator forward step.
                '''
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator(
                        x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, temperature
                    )
                if t % step_size == 0:
                    real_goal = last_goal
                    last_goal = Variable(torch.zeros(
                        batch_size, goal_out_size
                    ))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                gen_token_list.append(x_t)
                t = t_
            gen_token = torch.stack(gen_token_list).permute(1, 0)
            return gen_token
        return func
    elif f_type == 'gen':
        def func(model_dict, use_cuda=False, temperature=1.0):
            '''
            Get generator and discriminator
            '''
            generator = model_dict['generator']
            discriminator = model_dict['discriminator']

            '''
            Initialize variables and lists for forward step.
            '''
            h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = \
                init_vars(generator, discriminator, use_cuda)
            t = 0
            gen_token_list = []
            batch_size = generator.worker.batch_size
            seq_len = discriminator.seq_len
            step_size = generator.step_size
            goal_out_size = generator.worker.goal_out_size
            vocab_size = discriminator.vocab_size

            '''
            Perform generator forward step.
            '''
            while t < seq_len:
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), vocab_size)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t), value=vocab_size
                    )
                f_t = discriminator(cur_sen)["feature"]

                '''
                Generator forward step.
                '''
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t_ = generator(
                        x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal,
                        real_goal, t, temperature
                    )
                if t % step_size == 0:
                    if t > 0:
                        real_goal = last_goal
                    last_goal = Variable(torch.zeros(
                        batch_size, goal_out_size
                    ))
                    if use_cuda:
                        last_goal = last_goal.cuda(async=True)
                gen_token_list.append(x_t)
                t = t_
            gen_token = torch.stack(gen_token_list).permute(1, 0)
            return gen_token
        return func
    else:
        raise ("Invalid function type!")

def get_sample(model_dict, use_cuda=False, temperature=1.0):
    return recurrent_func('gen')(model_dict, use_cuda, temperature)

def get_rewards(model_dict, input_x, rollout_num,
               use_cuda=False, temperature=1.0, delta=16.0):
    '''
    Get generator and discriminator.
    '''
    generator = model_dict["generator"]
    discriminator = model_dict["discriminator"]
    discriminator = discriminator.eval()

    '''
    Prepare some constants.
    '''
    seq_len = discriminator.seq_len
    step_size = generator.step_size

    '''
    Perform rollout and calculate reward.
    '''
    rewards = []
    rollout_func = recurrent_func('rollout')
    for i in range(rollout_num):
        given_num = 0
        while given_num < seq_len:
            sample_for_reward = rollout_func(model_dict, input_x, given_num,
                                             use_cuda, temperature)
            pred = discriminator(sample_for_reward)['pred']
            pred = pred[:, 1].data.numpy()
            pred = pred.reshape(-1)
            if i == 0:
                rewards.append(pred)
            else:
                rewards[int(given_num / step_size - 1)] += pred
            given_num += step_size
    rewards = rescale(rewards, delta) / rollout_num
    if use_cuda:
        rewards = rewards.cuda(async=True)
    discriminator = discriminator.train()
    return rewards

def rescale(rewards, delta=16.0):
    '''
    : param rewards:
        type: list
        length: seq_len / c
        elements: np.array(size=batch_size)
    '''
    r = np.array(rewards) # r [seq_len / c * batch_size]
    _, batch_size = r.shape
    order = np.argsort(r)
    rank = np.argsort(order)
    rank = batch_size - rank
    rescaled_rewards = expit(delta * (0.5 - rank / batch_size))
    rescaled_rewards = np.transpose(rescaled_rewards)
    return Variable(torch.from_numpy(rescaled_rewards)).float()

def one_hot(x, vocab_size, use_cuda=False):
    batch_size, seq_len = x.size()
    out = torch.zeros(batch_size * seq_len, vocab_size)
    x = x.contiguous()
    x = x.view(-1, 1)
    out = out.scatter_(1, x.data, 1.0)
    out = out.view(batch_size, seq_len, vocab_size)
    out = Variable(out)
    if use_cuda:
        out = out.cuda(async=True)
    return out

def loss_func(f_type='pre_worker'):
    '''
    There are five kinds of loss functions:
        'pre_worker', 'pre_manager', 'adv_worker', 'adv_manager',
        'dis'
    '''

    if f_type == 'pre_manager':
        def func(real_goal, delta_feature):
            loss = torch.mean(1.0 - F.cosine_similarity(
                real_goal, delta_feature, dim=2
            ))
            return -loss
        return func

    elif f_type == 'pre_worker':
        def func(real_data, prediction, vocab_size, use_cuda=False):
            prediction = torch.clamp(prediction, 1e-20, 1.0)
            loss = -torch.mean(
                one_hot(real_data, vocab_size, use_cuda) *
                torch.log(prediction)
            )
            return loss
        return func

    elif f_type ==  'adv_manager':
        def func(rewards, real_goal, delta_feature):
            loss = -torch.mean(
                rewards * (1.0 - F.cosine_similarity(
                    delta_feature, real_goal, dim=2
                ))
            )
            return loss
        return func

    elif f_type == 'adv_worker':
        def func(all_goal, delta_feature_for_worker, gen_token,
                 prediction, vocab_size, use_cuda=False):
            intrinsic_rewards = 1.0 - F.cosine_similarity(
                all_goal, delta_feature_for_worker, dim=2
            )
            prediction = torch.clamp(prediction, 1e-20, 1.0)
            loss = -torch.mean(intrinsic_rewards * torch.sum(
                one_hot(gen_token, vocab_size, use_cuda) *
                torch.log(prediction)
            , dim=2))
            return loss
        return func

    elif f_type == 'dis':
        def func(discriminator, input_x, score, use_cuda=False):
            '''
            : param input_x:
                size(batch_size * seq_len)
                type(torch.LongTensor)
            : param score:
                size(batch_size * seq_len * vocab_size)
                type(torch.FloatTensor)
            '''
            loss_func = nn.CrossEntropyLoss()
            if use_cuda:
                loss_func = loss_func.cuda()
            input_x = input_x.view(-1)
            batch_size, seq_len, out_size = score.size()
            score = score.view(batch_size * seq_len, -1)
            loss = loss_func(score, input_x) + discriminator.l2_loss()
            return loss
        return func

    else:
        raise ("Invalid loss function type!")
