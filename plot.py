import math
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def constrains_echo_space(t2):

    for i in range(3):
        t2[i] = 0

    return t2



def plot_true_data_fig(model_path, echo_num, echo_space, echo_noise_train, denoising_echo, t2_prediction):
    # Plotting the echo
    t_list = []
    for i in range(echo_num):
        t_list.append((i + 1) * echo_space)
    t = np.array(t_list).reshape(2500, 1)
    echo_noise_train = np.array(echo_noise_train).reshape(2500, 1)
    denoising_echo = np.array(denoising_echo).reshape(2500, 1)


    plt.figure()
    plt.xlim(0, 500)
    plt.ylim(-0.5, 2)
    plt.xlabel('Time(ms)')
    plt.ylabel('Echo')
    plt.plot(t, denoising_echo, label='denoising_echo', zorder=3, color='red')
    plt.plot(t, echo_noise_train, label='echo_noise', zorder=2, color='black')
    plt.legend()
    plt.axis()
    # plt.savefig(model_path + "\\fig\\" + str(num) + '_Echo')
    plt.show()


    t2_smooth = scipy.signal.savgol_filter(t2_prediction, 20, 3)
    t2_smooth = pd.DataFrame(t2_smooth)
    t2_smooth = constrains_echo_space(t2_smooth)
    t2_smooth[t2_smooth < 0] = 0
    t2_smooth.to_csv(model_path + "/t2_smooth.csv", header=False, index=False)
    t2_smooth = np.array(t2_smooth).reshape(128, 1)


    T2 = np.logspace(np.log10(0.1), np.log10(10000), 128)
    t2_prediction = np.array(t2_prediction).reshape(128, 1)





    plt.xlabel('T2 (ms)')
    plt.ylabel('Amplitude (p.u.)')

    plt.plot(T2, t2_prediction, label='T2 prediction')
    plt.plot(T2, t2_smooth, label='T2 Smooth')
    plt.xscale('log')
    # plt.xlim(0, 10000)
    plt.legend()
    plt.axis()
    plt.savefig(model_path + "\\fig\\" + str(num) + '_T2.png')
    plt.show()


def plot_inverse_data_fig(echo_num, echo_space, echo, echo_noise, denoising_echo, T2_AMPLITUDE, t2, model_path):
    # plot echo and specta
    t_list = []
    for i in range(echo_num):
        t_list.append((i + 1) * echo_space)
    t = np.array(t_list).reshape(2500, 1)
    echo = np.array(echo).reshape(2500, 1)
    echo_noise = np.array(echo_noise).reshape(2500, 1)
    denoising_echo = np.array(denoising_echo).reshape(2500, 1)


    plt.figure()
    plt.xlim(0, 500)
    plt.ylim(-0.5, 12)
    plt.xlabel('Time(ms)')
    plt.ylabel('Echo')
    plt.plot(t, echo, label='Pure echo', zorder=3, color='red')
    plt.plot(t, denoising_echo, label='denoisng echo', zorder=2, color='black')
    plt.plot(t, echo_noise, label='noisy echo', zorder=1, color='yellow')
    plt.legend()
    plt.axis()
    plt.savefig(model_path + '/Echo')
    plt.show()

    T2 = np.logspace(np.log10(0.1), np.log10(10000), 128)
    T2 = np.array(T2).reshape(128, 1)


    t2_result = scipy.signal.savgol_filter(t2, 20, 3)
    t2_result = scipy.signal.savgol_filter(t2_result, 10, 3)
    t2_result = pd.DataFrame(t2_result)
    t2_result[t2_result < 0] = 0
    t2_result = np.array(t2_result).reshape(128, 1)
    t2 = pd.DataFrame(t2)

    t2 = np.array(t2).reshape(128, 1)
    T2_AMPLITUDE = np.array(T2_AMPLITUDE).reshape(128, 1)


    plt.xlabel('T2 (ms)')
    plt.ylabel('Amplitude (p.u.)')
    T2_AMPLITUDE = T2_AMPLITUDE
    plt.plot(T2, T2_AMPLITUDE, label='AMPLITUDE')
    plt.plot(T2, t2_result, label='T2 LMsNN')


    plt.xscale('log')
    plt.legend()
    plt.axis()
    plt.savefig(model_path + '/T2.png')
    plt.show()

    return t2_result












