from torch.autograd import Variable
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
    ), discriminator.start_token))
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

            '''
            Perform forward step for pretraining generator and
            dicriminator.
            '''
            while t < seq_len + 1 :
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), -1)
                    ).long()
                else:
                    cur_sen = real_data[:, :t]
                    cur_sen = cur_sen.contiguous()
                    cur_sen = F.pad(
                        cur_sen.view(-1, t), (0, seq_len - t), value=-1
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
            prediction_var = torch.stack(prediction_list).permuet(1, 0, 2)
            delta_feature_var = torch.stack(delta_feature_list).permute(1, 0, 2)
            rets = {"real_goal":real_goal_var,
                    "prediction":prediction_var,
                    "delta_feature":delta_feature_var}
            for ret in rets.values():
                if ret.is_contiguous():
                    ret = ret.contiguous()
            return rets

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
                        torch.zeros(batch_size, seq_len), -1)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t), value=-1
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
                        delta_feature_for_worker_list.append(
                            f_t - feature_list[t - step_size]
                        )
                    else:
                        delta_feature_for_worker_list.append(
                            f_t - feature_list[t - t%step_size]
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
            real_goal_var = torch.stack(real_goal_list).permute(1, 0, 2)
            all_goal_var = torch.stack(all_goal_list).permute(1, 0, 2)
            prediction_var = torch.stack(prediction_list).permuet(1, 0, 2)
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

            '''
            Use input_x to perform generator forward step.
            '''
            while t < given_num + 1:
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len), -1)
                    ).long()
                    if use_cuda:
                        cur_sen = cur_sen.cuda(async=True)
                else:
                    cur_sen = torch.stack(gen_token_list).permute(1, 0)
                    cur_sen = F.pad(
                        cur_sen, (0, seq_len - t), value=-1
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
                if t > 0:
                    gen_token_list.append(
                        input_x[:, t - 1]
                    )
                t = t_

            '''
            Perform rollout.
            '''
            while t < seq_len + 1:

                '''
                Extract feature f_t.
                '''
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                cur_sen = F.pad(
                    cur_sen, (0, seq_len - t), value=-1
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

    else:
        raise ("Invalid function type!")
