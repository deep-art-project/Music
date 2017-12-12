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
                init_vars(generator,discriminator, use_cuda)
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
            while t < discriminator.seq_len + 1 :
                '''
                Extract feature f_t.
                '''
                if t == 0:
                    cur_sen = Variable(nn.init.constant(
                        torch.zeros(batch_size, seq_len)
                    ), -1).long()
                else:
                    cur_sen = real_data[:, :t]
                    cur_sen = cur_sen.contiguous()
                    cur_sen = F.pad(
                        cur_sen.view(-1, t + 1), (0, seq_len - t - 1), value=-1
                    )
                if use_cuda:
                    cur_sen = cur_sen.cuda(async=True)
                f_t = discriminator(cur_sen)["feature"]

                '''
                Generator forward step.
                '''
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal,\
                sub_goal, probs, t = generator(
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

            '''
            Post process and return variables needed for calculating loss.
            '''
            if len(real_goal_list) == len(delta_feature_list) + 1:
                real_goal_list = real_goal_list[:-1]
            prediction_list = prediction_list[:-1]
            real_goal = torch.stack(real_goal_list).permute(1, 0, 2)
            prediction = torch.stack(prediction_list).permuet(1, 0, 2)
            delta_feature = torch.stack(delta_feature_list).permute(1, 0, 2)
            rets = [real_goal, prediction, delta_feature]
            for ret in rets:
                if ret.is_contiguous():
                    ret = ret.contiguous()
            return rets

    elif f_type == 'adv':
        def func(model_dict, use_cuda=False, temperature=1.0):
            pass

    elif f_type == 'rollout':
        pass

    else:
        raise ("Invalid function type!")
