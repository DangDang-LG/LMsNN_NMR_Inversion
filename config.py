import numpy as np
import math

"""
Model 1
ratio = [0.6, 0.4]
SNR = 10
random_seed = 256
SNR = 20
random_seed = 75
SNR = 40
random_seed = 75
SNR = 80
random_seed = 65
"""

"""
Model 2
ratio = [0.4, 0.6]
SNR = 10
random_seed = 65
SNR = 20
random_seed = 75
SNR = 40
random_seed = 75
SNR = 80
random_seed = 75
"""

# Number of T2 points
T2_number = 128
# Number of echoes
echo_num = 2500
echo_space = 0.2
# signal-to-noise ratio
SNR = 10
# ratio of components
ratio = [0.4, 0.6]
T2_min = 0.1
T2_max = 10000
T2 = np.logspace(math.log10(T2_min), math.log10(T2_max), T2_number)

position_list = [10, 150]
sigma = [0.06, 0.06]

random_seed = 256
BATCH_SIZE = 1
EPOCHS = 12000


PROJ_DIR = './'
model_dir = './out_result/trained_models'
model_name = ''
model_path = model_dir + "/" + model_name

generated_pure_echo_path = "./forward_data/pure_echo/"
generated_echo_noise_path = "./forward_data/echo_noise/"
generated_T2_AMPLITUDE_path = "./forward_data/T2_AMPLITUDE/"

save_pure_echo_path = './forward_data/'
save_echo_noise_path = './forward_data/'
save_T2_AMPLITUDE_path = './forward_data/'

true_data_path = ''

