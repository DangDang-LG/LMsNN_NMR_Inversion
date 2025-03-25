import numpy as np
import math
from plot import plot_true_data_fig
from config import *
import os



def get_matrix_a(echo_num, echo_space, t2_sample_num, t2_min, t2_max):

    t_list = []

    for i in range(echo_num):
        t_list.append((i + 1) * echo_space)
    t = np.array(t_list)
    t = np.expand_dims(t, axis=-1)
    log_min, log_max = math.log10(t2_min), math.log10(t2_max)
    t2 = np.logspace(log_min, log_max, t2_sample_num, base=10)
    t2 = np.expand_dims(t2, axis=0)
    t2 = 1 / t2
    matrix_a = np.dot(t, t2)
    matrix_a = np.exp(-1 * matrix_a)
    return matrix_a


def normal_distribution(x, mean, sigma):

    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))




def forward_T2_inverse(ratio, T2_number, position, sigma, SNR, random_seed):

    # Echo spacing
    TE = 0.2
    # Number of echo
    NE = 2500
    # porosity
    por = 10
    # bondage water
    por_1 = por * ratio[0]
    # free water
    por_2 = por * ratio[1]
    # T2 minimum time
    T2_min = 0.1
    # T2 maximum time
    T2_max = 10000
    # T2 distribution point
    T2 = np.logspace(np.log10(T2_min), np.log10(T2_max), T2_number)
    x = np.linspace(0, 1, T2_number)

    position_1 = (math.log10(position[0])-math.log10(0.1))/(math.log10(10000)-math.log10(0.1))
    position_2 = (math.log10(position[1])-math.log10(0.1))/(math.log10(10000)-math.log10(0.1))

    Model_1 = normal_distribution(x, position_1, sigma[0])
    Model_2 = normal_distribution(x, position_2, sigma[1])

    f1 = Model_1 * por_1 / sum(Model_1)
    f2 = Model_2 * por_2 / sum(Model_2)

    # Verify if the porosity is 10
    f = f1 + f2
    f = f.reshape((T2_number, 1))
    A = get_matrix_a(NE, TE, T2_number, T2_min, T2_max)
    x = np.dot(A, f)
    deta = x[0] / SNR

    np.random.seed(random_seed)
    noise = np.random.normal(loc=0, scale=1, size=(NE, 1)) * deta
    y = x + noise
    f = f.reshape((1, T2_number))
    x_return = x[0:2500]
    y_return = y[0:2500]
    x_return = np.array(x_return).reshape(-1, 2500)
    y_return = np.array(y_return).reshape(-1, 2500)
    f = np.array(f).reshape(-1, 128)

    return x_return, y_return, f



