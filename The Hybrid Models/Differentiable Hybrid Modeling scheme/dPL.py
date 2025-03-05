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
tf.compat.v1.experimental.output_all_intermediates(True)

## Import libraries developed by this study
from dPL_class import Differentiable_EXPHYDRO, Differentiable_Xinanjiang, LSTM_postprocess

import loss

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = 'Your path'
attrs_path = 'Your path'#Replace the local path with the basin attributes data
mets_path = 'Your path'#Replace the local path with the training data

basin_id = []#The number of the regional training basin


# Define the start and end dates
start_date = pd.to_datetime('1975/10/1')
end_date = pd.to_datetime('1995/9/30')
all_list = []
for i in range(len(basin_id)):
    a = basin_id[i]
#Read meteorological forcing data
    curr_csv_path = f'{mets_path}\\{str(a)}_metv.csv'
    curr_basin = pd.read_csv(curr_csv_path, parse_dates=['date'], index_col='date')

    # total_precipitation_sum
    p = curr_basin['total_precipitation_sum_sum']
    subp = p[start_date:end_date]
    np_subp = subp.to_numpy().reshape((-1, 1)) #
    result_p = np_subp

    # temperature_2m
    tm = curr_basin['temperature_2m']
    subtm = tm[start_date:end_date]
    np_subtm = subtm.to_numpy().reshape((-1, 1))   #
    result_p_tm = np.concatenate((result_p, np_subtm), axis=1)

    # potential_evaporation_sum
    pet = curr_basin['potential_evaporation_sum']
    subpet = pet[start_date:end_date]
    np_subpet = subpet.to_numpy().reshape((-1, 1))*1000  #Convert units
    result_p_tm_pet = np.concatenate((result_p_tm, np_subpet), axis=1)

    # surface_pressure
    vp = curr_basin['surface_pressure']
    subvp = vp[start_date:end_date]
    np_subvp = subvp.to_numpy()
    np_subvp = np_subvp.reshape((-1, 1))/1000  #Convert units
    result_p_tm_pet_vp = np.concatenate((result_p_tm_pet, np_subvp), axis=-1)

    # surface_solar_radiation_downwards_sum
    srad = curr_basin['surface_solar_radiation_downwards_sum']
    subsrad = srad[start_date:end_date]
    np_subsrad = subsrad.to_numpy()
    np_subsrad = np_subsrad.reshape((-1, 1))/86400  #Convert units
    result_p_tm_pet_vp_srad = np.concatenate((result_p_tm_pet_vp, np_subsrad), axis=-1)

#Read basin attribute data
    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('basin_id')
    rows_bool = (static_x.index == str(basin_id[i]))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)

    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)

    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(result_p_tm_pet_vp_srad.shape[0], axis=0)

    result_p_tm_pet_vp_srad_xs = np.concatenate((result_p_tm_pet_vp_srad, local_static_x_for_train), axis=-1)

    # Q
    q = curr_basin['runoff_sum_v']
    subq = q[start_date:end_date]
    np_subq = subq.to_numpy()
    np_subq = np_subq.reshape((np_subq.shape[0], 1))

    result_p_tm_pet_vp_srad_xs_q = np.concatenate((result_p_tm_pet_vp_srad_xs, np_subq), axis=-1)

    all_list.append(result_p_tm_pet_vp_srad_xs_q)


print(len(all_list))
sum_result = all_list[0]

for i in range(len(all_list)-1):
    sum_result = np.concatenate((sum_result, all_list[i+1]), axis=0)
    print(sum_result.shape)

nan_rows = np.isnan(sum_result).any(axis=1)
sum_result = sum_result[~nan_rows]  #Check to eliminate nan values
sum_result1 = sum_result

#Data preprocessing
def generate_train_test(train_set, train_set1, wrap_length):
    train_set_ = pd.DataFrame(train_set)
    train_x_np = train_set_.values[:, :-1]

    print("p_mean:", np.mean(train_x_np[:, 0:1]))
    print("tmean_mean:", np.mean(train_x_np[:, 1:2]))
    print("pet_mean:", np.mean(train_x_np[:, 2:3]))
    print("vp_mean:", np.mean(train_x_np[:, 3:4]))
    print("srad_mean:", np.mean(train_x_np[:, 4:5]))
    print("p_std:", np.std(train_x_np[:, 0:1]))
    print("tmean_std:", np.std(train_x_np[:, 1:2]))
    print("pet_std:", np.std(train_x_np[:, 2:3]))
    print("vp_std:", np.std(train_x_np[:, 3:4]))
    print("srad_std:", np.std(train_x_np[:, 4:5]))

    train_set1_ = pd.DataFrame(train_set1)
    train_x_np1 = train_set1_.values[:, :-1]
#It should be adjusted according to the actual training data
    train_x_np1[:,0:1] = (train_x_np1[:,0:1] - 2.3911780537521694)/6.116518250993573
    train_x_np1[:,1:2] = (train_x_np1[:,1:2] - 8.544640829988298)/12.797279204185681
    train_x_np1[:,2:3] = (train_x_np1[:,2:3] - 4.662763932110111)/4.015221740698475
    train_x_np1[:,3:4] = (train_x_np1[:,3:4] - 86.79319736735056)/15.068813725328999
    train_x_np1[:,4:5] = (train_x_np1[:,4:5] - 178.64857170578009)/82.53047463859957

    train_y_np1 = train_set1_.values[:, -1:]
    print("q_mean:",  np.mean(train_y_np1[:,-1:]))
    print("q_std:",  np.std(train_y_np1[:,-1:]))

    wrap_number_train = (train_x_np.shape[0] - wrap_length) // 31 + 1

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_x1 = np.empty(shape=(wrap_number_train, wrap_length, train_x_np1.shape[1]))
    train_y1 = np.empty(shape=(wrap_number_train, wrap_length, train_y_np1.shape[1]))


    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 31:(wrap_length + i * 31), :]
        train_x1[i, :, :] = train_x_np1[i * 31:(wrap_length + i * 31), :]
        train_y1[i, :, :] = train_y_np1[i * 31:(wrap_length + i * 31), :]

    return train_x, train_x1, train_y1

wrap_length = 365  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
train_x, train_x1, train_y = generate_train_test(sum_result, sum_result1, wrap_length=wrap_length)

print(f'The shape of train_x, train_x1, train_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_x1.shape}, {train_y.shape}')

#Data-driven models are coupled to different process-based models
def create_model(input_xd_shape, input_xd_shape1, seed):
    # batch_size must be adjusted according to wrap_length data
    xd_input_forprnn = Input(shape=input_xd_shape, batch_size=580, name='Input_xd')
    xd_input_forprnn1 = Input(shape=input_xd_shape1, batch_size=580, name='Input_xd1')

    hydro_output = Differentiable_EXPHYDRO(mode='normal', name='Regional_dPL_LSTM')(xd_input_forprnn)

    xd_hydro = Concatenate(axis=-1, name='Concat')([xd_input_forprnn1, hydro_output])

    e_hn, e_cn = LSTM_postprocess(input_xd = 21, hidden_size=256, seed=seed, name='LSTM')(xd_hydro)
    fc2_out = Dense(units=1)(e_hn)
    fc2_out = K.permute_dimensions(fc2_out, pattern=(1,0,2))  # for test model

    model = Model(inputs=[xd_input_forprnn,xd_input_forprnn1], outputs=fc2_out)
    return model


def train_model(model, train_xd, train_xd1, train_y, ep_number, lrate, save_path):
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)

    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)

    tnan = callbacks.TerminateOnNaN()

    model.compile(loss=loss.nse_loss, metrics=[loss.nse_metrics],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))

    history = model.fit(x=[train_xd, train_xd1], y=train_y, epochs=ep_number, batch_size=580,
                        callbacks=[save, es, reduce, tnan])
    return history


save_path = f'your path'#Set the path for storing the model

model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]), input_xd_shape1=(train_x1.shape[1], train_x1.shape[2]), seed= 200)
model.summary()

train_history = train_model(model=model, train_xd=train_x,train_xd1=train_x1, train_y=train_y, ep_number=150, lrate=0.01, save_path=save_path)
