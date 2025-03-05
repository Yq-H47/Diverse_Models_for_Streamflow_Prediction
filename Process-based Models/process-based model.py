import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf

## Import libraries developed by this study
from cn_class import local_EXPHYDRO, local_Xinanjiang
import loss

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = 'Your path'
attrs_path = 'Your path'#Replace the local path with the basin attributes data
mets_path = 'Your path'#Replace the local path with the training data

basin_id = []#The number of the regional training basin

for i in range(len(basin_id)):
    K.clear_session()

    def handle_invalid_values(data):
        #mask = np.isnan(data) | np.isinf(data)
        array_2d = np.nan_to_num(data, nan=0.0)
        return array_2d

    def normalize_minmax(data):
        data_min = np.min(data)
        print("data_min:",data_min)
        data_max = np.max(data)
        data_scaled = (data - data_min) / (data_max - data_min)
        return data_scaled

    def normalize(data):
        train_mean = np.mean(data, axis=0, keepdims=True)
        train_std = np.std(data, axis=0, keepdims=True)
        train_scaled = (data - train_mean) / train_std
        return train_scaled

 # Define the start and end dates
    start_date = pd.to_datetime('1975/10/1')
    end_date = pd.to_datetime('1995/9/30')

    a = basin_id[i]
    print(a)
#Read meteorological data
    curr_csv_path = f'{mets_path}\\{str(a)}_metv.csv'
    curr_basin = pd.read_csv(curr_csv_path, parse_dates=['date'], index_col='date')

# potential_evaporation_sum
    pet = curr_basin['potential_evaporation_sum']
    subpet = pet[start_date:end_date]
    np_subpet = subpet.to_numpy().reshape((-1, 1))*1000 #Convert units
    result_pet = np_subpet

# temperature_2m
    tm = curr_basin['temperature_2m']
    subtm = tm[start_date:end_date]
    np_subtm = subtm.to_numpy().reshape((-1, 1))
    result_pet_tm = np.concatenate((result_pet, np_subtm), axis=1)

# total_precipitation_sum
    p = curr_basin['total_precipitation_sum']
    subp = p[start_date:end_date]
    np_subp = subp.to_numpy().reshape((-1, 1))*1000 #Convert units
    result_pet_tm_p = np.concatenate((result_pet_tm, np_subp), axis=1)

 # surface_pressure
    vp = curr_basin['surface_pressure']
    subvp = vp[start_date:end_date]
    np_subvp = subvp.to_numpy()
    np_subvp = np_subvp.reshape((-1, 1))/1000 #Convert units
    result_pet_tm_p_vp = np.concatenate((result_pet_tm_p, np_subvp), axis=-1)

# surface_solar_radiation_downwards_sum
    srad = curr_basin['surface_solar_radiation_downwards_sum']
    subsrad = srad[start_date:end_date]
    np_subsrad = subsrad.to_numpy()
    np_subsrad = np_subsrad.reshape((-1, 1))/86400 #Convert units
    result_pet_tm_p_vp_srad = np.concatenate((result_pet_tm_p_vp, np_subsrad), axis=-1)


# This is not required when reading basin attribute data for local modeling
    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('Basin')
    rows_bool = (static_x.index == str(basin_id[i]))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    print("!!!",rows_list)
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)

    local_static_x = static_x_np[rows_int, :]
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)

    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(result_pet_tm_p_vp_srad.shape[0], axis=0)

    result_pet_tm_p_vp_srad_xs = np.concatenate((result_pet_tm_p_vp_srad, local_static_x_for_train), axis=-1)

# Q
    q = curr_basin['runoff_sum_v']
    subq = q[start_date:end_date]
    np_subq = subq.to_numpy()
    np_subq = np_subq.reshape((np_subq.shape[0], 1))

    result_pet_tm_p_vp_srad_xs_q = np.concatenate((result_pet_tm_p_vp_srad_xs, np_subq), axis=-1)
    sum_result = result_pet_tm_p_vp_srad_xs_q

    nan_rows = np.isnan(sum_result).any(axis=1)
    sum_result = sum_result[~nan_rows]

    def generate_train_test(train_set, wrap_length):
        train_set_ = pd.DataFrame(train_set)
        train_x_np = train_set_.values[:, :-1]

        train_y_np = train_set_.values[:, -1:]

        wrap_number_train = (train_x_np.shape[0] - wrap_length) // 3 + 1

        train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
        train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

        for i in range(wrap_number_train):
            train_x[i, :, :] = train_x_np[i * 3:(wrap_length + i * 3), :]
            train_y[i, :, :] = train_y_np[i * 3:(wrap_length + i * 3), :]

        return train_x, train_y

    wrap_length = 270  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
    train_x, train_y = generate_train_test(sum_result, wrap_length=wrap_length)

    print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
    print(f'{train_x.shape}, {train_y.shape}')

    def create_model(input_xd_shape):
        xd_input_forprnn = Input(shape=input_xd_shape, batch_size=782, name='Input_xd1')
        # batch_size must be adjusted according to wrap_length data
        hydro_output = local_EXPHYDRO(mode='normal', name='local_exphydro')(xd_input_forprnn)

        model = Model(inputs=xd_input_forprnn, outputs=hydro_output)
        return model

    def train_model(model, train_xd, train_y, ep_number, lrate, save_path):
        save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                         save_weights_only=True)

        es = callbacks.EarlyStopping(monitor='mse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                     restore_best_weights=True)

        reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                             min_delta=0.005, cooldown=0, min_lr=lrate / 100)

        tnan = callbacks.TerminateOnNaN()

        model.compile(loss=loss.nse_loss, metrics=[loss.nse_metrics],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))

        history = model.fit(x=train_xd, y=train_y, epochs=ep_number, batch_size=782,
                            callbacks=[save, es, reduce, tnan])
        return history



    save_path_ealstm = f'your path/{basin_id[i]}_exp.h5' #Set the path for storing the model

    model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]))
    model.summary()

    train_history = train_model(model=model, train_xd=train_x,
                                      train_y=train_y, ep_number=200, lrate=0.01, save_path=save_path_ealstm)
    del model
