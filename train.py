import math
import os
import time
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from forward import forward_T2_inverse
from config import *
from plot import plot_inverse_data_fig
from network import Custom_Loss_Layer






def training_inverse(network_model, random_seed):
    start_time = time.time()
    model_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ' NMR_T2_Analysis'
    save_model_path = os.path.join(PROJ_DIR, 'out_result', 'trained_models', model_name)
    pure_echo_train, echo_noise_train, T2_AMPLITUDE_train = forward_T2_inverse(ratio, T2_number, position_list, sigma, SNR, random_seed)


    nmr_model = network_model()
    print('training...')
    history = nmr_model.fit([echo_noise_train],[echo_noise_train],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1
                            )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')

    plt.legend()
    plt.show()
    nmr_model.save(os.path.join(save_model_path, 'NMR_T2_Analysis'),save_format='tf')
    end_time = time.time()
    print("Training time:{:0.2f}s".format(end_time - start_time))
    return save_model_path



def predict_inverse(model_path, random_seed):

    model = load_model(model_path + "/NMR_T2_Analysis", custom_objects={"Custom_Loss_Layer": Custom_Loss_Layer})
    print(model.summary())
    pure_echo_train, echo_noise_train, T2_AMPLITUDE_train = forward_T2_inverse(ratio, T2_number, position_list, sigma, SNR, random_seed)
    result = model.predict([echo_noise_train])
    denoising_echo = result
    t2_model = Model(inputs=model.input, outputs=model.get_layer('t2').output)
    t2_prediction = t2_model.predict(echo_noise_train)

    pure_echo_train = pure_echo_train.reshape(-1, 2500)
    denoising_echo = denoising_echo.reshape(-1, 2500)
    echo_noise_train = echo_noise_train.reshape(-1, 2500)
    t2_prediction = t2_prediction.reshape(-1, 128)
    T2_AMPLITUDE_train = T2_AMPLITUDE_train.reshape(-1, 128)

    # smooth result
    t2_smooth = plot_inverse_data_fig(echo_num, echo_space, pure_echo_train, echo_noise_train, denoising_echo, T2_AMPLITUDE_train, t2_prediction, model_path)
    t2_smooth = t2_smooth.reshape(-1, 128)


    pure_echo_train = pd.DataFrame(pure_echo_train)
    denoising_echo = pd.DataFrame(denoising_echo)
    echo_noise_train = pd.DataFrame(echo_noise_train)
    t2_prediction = pd.DataFrame(t2_prediction)
    # t2_prediction[t2_prediction < 0] = 0
    T2_AMPLITUDE_train = pd.DataFrame(T2_AMPLITUDE_train)
    t2_smooth = pd.DataFrame(t2_smooth)

    # calculate T2
    t2_mse = mean_squared_error(T2_AMPLITUDE_train, t2_smooth)
    t2_mae = mean_absolute_error(T2_AMPLITUDE_train, t2_smooth)
    t2_rmse = np.sqrt(mean_squared_error(T2_AMPLITUDE_train, t2_smooth))
    t2_r2 = r2_score(T2_AMPLITUDE_train, t2_smooth)

    # calculate RMSE
    # Calculate the echo signal
    echo_mse = mean_squared_error(pure_echo_train, denoising_echo)
    echo_mae = mean_absolute_error(pure_echo_train, denoising_echo)
    echo_rmse = np.sqrt(mean_squared_error(pure_echo_train, denoising_echo))
    echo_r2 = r2_score(pure_echo_train, denoising_echo)
    # Save
    t2_smooth.to_csv(model_path + "/t2_LMsNN.csv", header=False, index=False)
    T2_AMPLITUDE_train.to_csv(model_path + "/T2_AMPLITUDE.csv", header=False, index=False)

    denoising_echo.to_csv(model_path + "/denoising_echo.csv", header=False, index=False)
    pure_echo_train.to_csv(model_path + "/pure_echo.csv", header=False, index=False)
    echo_noise_train.to_csv(model_path + "/echo_noise_train.csv", header=False, index=False)
    # Output evaluation metrics
    print('echo mse: ' + str(echo_mse) + '  echo mae: ' + str(echo_mae) + ' echo rmse: ' + str(echo_rmse) + ' echo r2: ' + str(echo_r2))
    print('t2 mse: ' + str(t2_mse) + '  t2 mae: ' + str(t2_mae) + ' t2 rmse: ' + str(t2_rmse) + ' t2 r2: ' + str(t2_r2))
