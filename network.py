import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Layer, Attention, GlobalAvgPool1D, Bidirectional, Conv1D, Flatten, MaxPooling1D, Activation, Dot, Multiply, Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
# from keras_self_attention import SeqSelfAttention
from config import *


def get_matrix_constrain(t2_sample_num):

    array_1 = K.zeros((1, 13))
    initial_array = K.ones((1, 115))
    constrain_array = K.concatenate([array_1, initial_array], axis=-1)

    return constrain_array


def get_matrix_well_logging_data_constrain(t2_sample_num):

    array_1 = K.zeros((1, 3))
    initial_array = K.ones((1, 27))
    constrain_array = K.concatenate([array_1, initial_array], axis=-1)

    return constrain_array



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



# Customized Loss Functions
class Custom_Loss_Layer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(Custom_Loss_Layer, self).__init__(**kwargs)


    def my_loss(self, echo_noise, t2, denosing_echo):

        echo_loss = K.sqrt(K.sum(abs(echo_noise - denosing_echo), axis=-1))


        # Smooth constraints
        t2_smooth_limit = K.sqrt(K.sum(K.square(t2), axis=-1))
        loss_result = echo_loss + 0.2 * t2_smooth_limit

        return loss_result

    def call(self, inputs, **kwargs):
        echo_noise = inputs[0]
        t2 = inputs[1]
        denosing_echo = inputs[2]

        loss = self.my_loss(echo_noise, t2, denosing_echo)
        self.add_loss(loss, inputs=inputs)

        return denosing_echo




def nmr_inversion_model_cnn():

    input_shape_echo = (2500,)

    echo_noise = Input(shape=input_shape_echo)

    x = Dense(units=2500, activation='relu')(echo_noise)
    x = tf.expand_dims(x, axis=1)
    cov1 = Conv1D(filters=32, kernel_size=5, input_shape=input_shape_echo, strides=1, padding='same', name="cov1")(x)
    cov2 = Conv1D(filters=32, kernel_size=50, input_shape=input_shape_echo, strides=10, padding='same', name="cov2")(x)
    flat1 = Flatten()(cov1)
    flat2 = Flatten()(cov2)
    flat = K.concatenate([flat1, flat2])
    x = Dense(units=1024)(flat)
    x = Dense(units=512)(x)
    t2_1 = Dense(units=128, name='t2_1', use_bias=True, activation='softplus')(x)


    A = get_matrix_a(echo_num, echo_space, T2_number, T2_min, T2_max)
    A = K.variable(value=A, dtype='float32', name='A')
    A = K.permute_dimensions(A, (1, 0))
    constrain_array = get_matrix_constrain(T2_number)

    t2 = Multiply(name='t2')([t2_1, constrain_array])

    denosing_echo = K.dot(t2, A)

    output = Custom_Loss_Layer()([echo_noise, t2, denosing_echo])

    model = Model([echo_noise], [output])
    optimizer = Adam(learning_rate=0.0001,
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=1e-07,)
    model.compile(optimizer=optimizer,
                  loss=None,
                  metrics=["mse", "mae"])
    return model
