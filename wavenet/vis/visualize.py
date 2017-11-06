import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as plt


def get_loss(log_path):
    loss_list = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            loss_list.append(float(line.split(' ')[-1]))
    f.close()
    return loss_list


def plot_loss(log_path):
    loss = np.asarray(get_loss(log_path))
    fig = plt.figure()
    plt.plot(loss)
    fig.savefig('loss.png')


plot_loss('../log/loss_log.log')
