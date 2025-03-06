
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import Model, Sequential
from keras import optimizers, callbacks, losses, metrics
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, Input, Concatenate, Dropout
import keras.backend as K
import tensorflow as tf


from dPL_class import LSTM_postprocess
import loss

tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'


working_path = 'Your path'  # Your path
attrs_path = 'Your path'   # Static catchment attribute data path
met_path = 'Your path'    # Meteorological forcing data path
PBMQ_path= 'Your path'    # Path of runoff predicted based on process models


basin_id=[]  #Number of watershed participating in training

#Define start and end dates
start_date = pd.to_datetime('1975/10/1')
end_date = pd.to_datetime('1995/9/30')
all_list = []
for i in range(len(basin_id)):
    a = basin_id[i]

    curr_csv_path = f'{met_path}\\{str(a)}_metv.csv'
    PBMQ_patch_= f'{PBMQ_path}\\{str(a)}_runoff_predictions.csv'

    met_basin = pd.read_csv(curr_csv_path, parse_dates=['date'], index_col='date')
    PBMQ_basin = pd.read_csv(PBMQ_patch_, parse_dates=['date'], index_col='date')


# potential_evaporation_sum
    pet = met_basin['potential_evaporation_sum']
    subpet = pet[start_date:end_date]
    np_subpet = subpet.to_numpy().reshape((-1, 1))*1000  #Conversion unit
    result_pet = np_subpet

# temperature_2m
    tm = met_basin['temperature_2m']
    subtm = tm[start_date:end_date]
    np_subtm = subtm.to_numpy().reshape((-1, 1))  #
    result_pet_tm = np.concatenate((result_pet, np_subtm), axis=1)

# total_precipitation_sum
    p = met_basin['total_precipitation_sum']
    subp = p[start_date:end_date]
    np_subp = subp.to_numpy().reshape((-1, 1))*1000  #Conversion unit
    result_pet_tm_p = np.concatenate((result_pet_tm, np_subp), axis=1)

# surface_pressure
    vp = met_basin['surface_pressure']
    subvp = vp[start_date:end_date]
    np_subvp = subvp.to_numpy()
    np_subvp = np_subvp.reshape((-1, 1))/1000  #Conversion unit
    result_pet_tm_p_vp = np.concatenate((result_pet_tm_p, np_subvp), axis=-1)

# surface_solar_radiation_downwards_sum
    srad = met_basin['surface_solar_radiation_downwards_sum']
    subsrad = srad[start_date:end_date]
    np_subsrad = subsrad.to_numpy()
    np_subsrad = np_subsrad.reshape((-1, 1))/86400   #Conversion unit
    result_pet_tm_p_vp_srad = np.concatenate((result_pet_tm_p_vp, np_subsrad), axis=-1)

#Read runoff data predicted by process-based models
    PBMq = PBMQ_basin['pre_flow']
    subPBMq = PBMq[start_date:end_date]
    np_subPBMq = subPBMq.to_numpy()
    np_subPBMq = np_subPBMq.reshape((-1, 1))
    result_pet_tm_p_vp_srad = np.concatenate((result_pet_tm_p_vp_srad, np_subPBMq), axis=-1)

# Read basin attribute data
    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('basin_id')
    rows_bool = (static_x.index == str(basin_id[i]))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)

    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)

    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(result_pet_tm_p_vp_srad.shape[0], axis=0)

    result_pet_tm_p_vp_srad_xs = np.concatenate((result_pet_tm_p_vp_srad, local_static_x_for_train), axis=-1)

# Q
    q = met_basin['runoff_sum_v']
    subq = q[start_date:end_date]
    np_subq = subq.to_numpy()
    np_subq = np_subq.reshape((np_subq.shape[0], 1))
    result_pet_tm_p_vp_srad_xs_q = np.concatenate((result_pet_tm_p_vp_srad_xs, np_subq), axis=-1)

    all_list.append(result_pet_tm_p_vp_srad_xs_q)


print(len(all_list))
sum_result = all_list[0]

for i in range(len(all_list)-1):
    sum_result = np.concatenate((sum_result, all_list[i+1]), axis=0)
    print(sum_result.shape)

nan_rows = np.isnan(sum_result).any(axis=1)
sum_result = sum_result[~nan_rows]

print("sum_result_shape", sum_result.shape)

print("P_mean_std:", np.mean(sum_result[:, 0:1]), np.std(sum_result[:, 0:1]))
print("Q_mean_std:", np.mean(sum_result[:, -1:]), np.std(sum_result[:, -1:]))

#Preprocessing training data
def generate_train_test(train_set, wrap_length):

    print("PET_mean_std:", np.mean(train_set[:,0:1]), np.std(train_set[:,0:1]))
    print("T_mean_std:", np.mean(train_set[:,1:2]), np.std(train_set[:,1:2]))
    print("P_mean_std:", np.mean(train_set[:,2:3]), np.std(train_set[:,2:3]))
    print("Vp_mean_std:", np.mean(train_set[:,3:4]), np.std(train_set[:,3:4]))
    print("Srad_mean_std:", np.mean(train_set[:,4:5]), np.std(train_set[:,4:5]))
    print("PBMq_mean_std:", np.mean(train_set[:, 5:6]), np.std(train_set[:, 5:6]))

    train_x  = train_set[:,:-1]
#Adjust according to the data of actual participation in training
    train_x[:,0:1] = (train_set[:,0:1] - 4.662763932101928)/4.015221740689102
    train_x[:,1:2] = (train_set[:,1:2] - 8.54464082998831)/12.797279204185717
    train_x[:,2:3] = (train_set[:,2:3] - 3.225605393306491)/6.987026484936114
    train_x[:,3:4] = (train_set[:,3:4] - 86.79319736735113)/15.068813725329514
    train_x[:,4:5] = (train_set[:,4:5] - 178.6485717057808)/82.53047463860122
    train_x[:,5:6] = (train_set[:,5:6] - 0.8752079847922787)/ 2.081200095156107

    train_y_np = train_set[:, -1:]
    train_y_np_nor = (train_y_np - 0.9159176611167262)/2.6000096617178103

    print("Q_mean_std:", np.mean(train_y_np), np.std(train_y_np))

    wrap_number_train = (train_set.shape[0] - wrap_length) // 31  + 1

    train_x1 = np.empty(shape=(wrap_number_train, wrap_length, train_x.shape[1]))
    train_y1 = np.empty(shape=(wrap_number_train, wrap_length, train_y_np_nor.shape[1]))

    for i in range(wrap_number_train):
        train_x1[i, :, :] = train_x[i * 31 :(wrap_length + i * 31 ), :]
        train_y1[i, :, :] = train_y_np_nor[i * 31 :(wrap_length + i * 31 ), :]

    return train_x1, train_y1

wrap_length = 365  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
train_x, train_y = generate_train_test(sum_result, wrap_length=wrap_length)

print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_y.shape}')


def create_model(input_xd_shape, seed):
    #batch_size is adjusted based on wrap_length
    xd_input_forprnn = Input(shape=input_xd_shape, batch_size=580, name='Input_xd1')
    hn = LSTM_postprocess(input_xd=21, hidden_size=256, seed=seed)(xd_input_forprnn)
    fc_x = Dropout(0.4)(hn)
    fc_out = Dense(units=1)(fc_x)
    print("fc_out.shape", fc_out.shape)
    model = Model(inputs=xd_input_forprnn, outputs=fc_out)
    return model

def train_model(model, train_xd, train_y, ep_number, lrate, save_path):
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)

    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)

    tnan = callbacks.TerminateOnNaN()

    model.compile(loss= loss.nse_loss, metrics=[loss.nse_metrics],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))

    history = model.fit(x=train_xd, y=train_y, epochs=ep_number, batch_size=580,
                        callbacks=[save, es, reduce, tnan])
    return history


Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_MODEL = f'{working_path}/MODEL_NAME'   #Model save path

model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]), seed=200)
model.summary()


p_history = train_model(model=model, train_xd=train_x,
                            train_y=train_y, ep_number=150, lrate=0.01, save_path=save_path_MODEL)



