import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

class Differentiable_EXPHYDRO(Layer):

    def __init__(self, mode='normal', h_nodes=256, seed=200, **kwargs):
        self.mode = mode
        self.h_nodes = h_nodes
        self.seed = seed
        super(Differentiable_EXPHYDRO, self).__init__(**kwargs)

    def build(self, input_shape):

        self.prnn_w1 = self.add_weight(name='prnn_w1',
                                       shape=(15, self.h_nodes),
                                       initializer=initializers.RandomUniform(seed=self.seed - 5),

                                       trainable=True)

        self.prnn_b1 = self.add_weight(name='prnn_b1',
                                       shape=(self.h_nodes,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w2 = self.add_weight(name='prnn_w2',
                                       shape=(self.h_nodes, 64),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b2 = self.add_weight(name='prnn_b2',
                                       shape=(64,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w3 = self.add_weight(name='prnn_w3',
                                       shape=(64, 6),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b3 = self.add_weight(name='prnn_b3',
                                       shape=(6,),
                                       initializer=initializers.zeros(),
                                       trainable=True)


        self.shape = input_shape

        super(Differentiable_EXPHYDRO, self).build(input_shape)

    def heaviside(self, x):

        return (K.tanh(5 * x) + 1) / 2
#Division of rain and snow
    def rainsnowpartition(self, p, t, tmin):

        tmin = tmin * -3

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]
#Snow melt calculation
    def snowbucket(self, s0, t, ddf, tmax):

        ddf = ddf * 5
        tmax = tmax  * 3

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

#Evapotranspiration calculation
    def soilbucket(self, s1, pet, f, smax, qmax):

        f = f / 10
        smax = smax * 1400 + 100
        qmax = qmax * 40 + 10

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)

        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        pet = step_in[:, 0:1]
        t = step_in[:, 1:2]
        p = step_in[:, 2:3]

        tmin = step_in[:, 3:4]
        tmax = step_in[:, 4:5]
        ddf  = step_in[:, 5:6]
        f    = step_in[:, 6:7]
        smax = step_in[:, 7:8]
        qmax = step_in[:, 8:9]

        [_ps, _pr] = self.rainsnowpartition(p, t, tmin)

        _m = self.snowbucket(s0, t, ddf, tmax)

        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, f, smax, qmax)

        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 2:3]
        tmean = inputs[:, :, 1:2]
        pet = inputs[:, :, 0:1]

        attrs = inputs[:,:,5:]

        paras = K.tanh(K.dot(attrs, self.prnn_w1)+ self.prnn_b1) # layer 1
        paras = K.tanh(K.dot(paras, self.prnn_w2)+ self.prnn_b2) # layer 2
        parameters = K.sigmoid(K.dot(paras, self.prnn_w3)+ self.prnn_b3) # layer 3

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((pet, tmean, prcp, parameters), axis=-1)

        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        tmin = parameters[:, :, 0:1]
        tmax = parameters[:, :, 1:2]
        ddf  = parameters[:, :, 2:3]
        f    = parameters[:, :, 3:4]
        smax = parameters[:, :, 4:5]
        qmax = parameters[:, :, 5:6]


# Calculate final process variables
        [psnow, prain] = self.rainsnowpartition(prcp, tmean, tmin)

        m = self.snowbucket(s0, tmean, ddf, tmax)

        [et, qsub, qsurf] = self.soilbucket(s1, pet, f, smax, qmax)

        q = qsub+qsurf

        if self.mode == "normal":
            print("NORMAL!!!")
            return  q
        elif self.mode == "analysis":
            return q

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 1)

class Differentiable_Xinanjiang(Layer):
    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(Differentiable_Xinanjiang, self).__init__(**kwargs)

    def build(self, input_shape):

        self.prnn_w1 = self.add_weight(name='prnn_w1',
                                       shape=(15, self.h_nodes),
                                       initializer=initializers.RandomUniform(seed=self.seed - 5),

                                       trainable=True)

        self.prnn_b1 = self.add_weight(name='prnn_b1',
                                       shape=(self.h_nodes,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w2 = self.add_weight(name='prnn_w2',
                                       shape=(self.h_nodes, 64),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b2 = self.add_weight(name='prnn_b2',
                                       shape=(64,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w3 = self.add_weight(name='prnn_w3',
                                       shape=(64, 6),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b3 = self.add_weight(name='prnn_b3',
                                       shape=(8,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        super(Differentiable_Xinanjiang, self).build(input_shape)

    def heaviside(self, x):
        return (K.tanh(10 * x) + 1) / 2

#Evapotranspiration calculation
    def evapotranspiration(self, wu, wl, wd, pet):
        et1 = self.heaviside(wu-pet)*pet + self.heaviside(pet-wu)*wu
        remaining_pet = self.heaviside(0 -(pet - et1))*0 + self.heaviside((pet - et1) - 0)*(pet - et1)
        et22 = self.heaviside(remaining_pet-wl)*wl + self.heaviside(wl-remaining_pet)*remaining_pet
        et2 = self.heaviside(remaining_pet) * et22
        et33 = self.heaviside((remaining_pet - et2)-wd)*wd + self.heaviside(wd-(remaining_pet - et2))*(remaining_pet - et2)
        et3 = self.heaviside(remaining_pet - et2) * et33
        return et1, et2, et3, pet - et1 - et2 - et3
#Flow production calculation
    def runoff_production(self, wu, wl, wd, p, wum, wlm, wdm, b, c):
        wum = wum * 19.9 + 0.1
        wlm = wlm * 30 + 60
        wdm = wdm * 60 + 60
        c = c * 0.19 + 0.01
        b = b * 0.3 + 0.1
        w_total = wum + wlm + wdm
        s = c * (wu / w_total) ** 2 + b * (wl / w_total) ** 2
        runoff = self.heaviside(p - s) * (p - s)
        return runoff

    def water_source_partition(self, runoff, k1, k2, k3):
        k1 = k1 * 0.69 + 0.01
        k2 = k2 * 0.69 + 0.01
        k3 = k3 * 0.89 + 0.01
        surface_runoff =k1 * runoff
        interflow = k2 * (runoff - surface_runoff)
        baseflow = k3 * (runoff - surface_runoff - interflow)
        return surface_runoff, interflow, baseflow
#routing calculation
    def flow_routing(self, surface_runoff, interflow, baseflow):
        routed_surface = surface_runoff
        routed_interflow = interflow * 0.5
        routed_baseflow = baseflow * 0.25
        total_q = routed_surface + routed_interflow + routed_baseflow
        return total_q

    def step_do(self, step_in, states):
        wu = states[0][:, 0:1]
        wl = states[0][:, 1:2]
        wd = states[0][:, 2:3]

        pet = step_in[:, 0:1]
        p = step_in[:, 1:2]

        wum = step_in[:, 3:4]
        wlm = step_in[:, 4:5]
        wdm  = step_in[:, 5:6]
        b    = step_in[:, 6:7]
        c = step_in[:, 7:8]
        k1 = step_in[:, 8:9]
        k2 = step_in[:, 9:10]
        k3 = step_in[:, 10:11]

        [_et1, _et2, _et3] = self.evapotranspiration(wu, wl, wd, pet)
        _dwu = p - _et1
        _dwl = p - _et1 - _et2
        _dwd = p - _et1 - _et2 - _et3
        wu_next = wu + K.clip(_dwu, -1e5, 1e5)
        wl_next = wl + K.clip(_dwl, -1e5, 1e5)
        wd_next = wd + K.clip(_dwd, -1e5, 1e5)
        #total_et = et1 + et2 + et3

        _runoff = self.runoff_production(wu_next, wl_next, wd_next, p, wum,wlm,wdm,b,c)

        [_surface_runoff, _interflow, _baseflow] = self.water_source_partition(_runoff,k1,k2,k3)

        step_out = [K.concatenate([wu_next, wl_next, wd_next], axis=1)]

        return step_out, [step_out]

    def call(self, inputs):

        pet = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        p = inputs[:, :, 2:3]
        attrs = inputs[:, :, 5:]
        paras = K.tanh(K.dot(attrs, self.prnn_w1)+ self.prnn_b1) # layer 1
        paras = K.tanh(K.dot(paras, self.prnn_w2)+ self.prnn_b2) # layer 2
        parameters = K.sigmoid(K.dot(paras, self.prnn_w3)+ self.prnn_b3) # layer 3

        new_inputs = K.concatenate((pet, p, tmean, parameters), axis=-1)

        init_states = [K.zeros((K.shape(new_inputs)[0], 3))]

        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)
        wu = outputs[:, :, 0:1]
        wl = outputs[:, :, 1:2]
        wd = outputs[:, :, 2:3]

        wum = parameters[:, :, 0:1]
        wlm = parameters[:, :, 1:2]
        wdm  = parameters[:, :, 2:3]
        b    = parameters[:, :, 3:4]
        c = parameters[:, :, 4:5]
        k1 = parameters[:, :, 5:6]
        k2 = parameters[:, :, 6:7]
        k3 = parameters[:, :, 7:8]

        [et1, et2, et3] = self.evapotranspiration(wu, wl, wd, pet)
        runoff = self.runoff_production(wu,wd,wl,p,wum,wdm,wlm,b,c)
        [surface_runoff,interflow,baseflow]=self.water_source_partition(runoff, k1, k2, k3)
        total_q = self.flow_routing(surface_runoff, interflow, baseflow)

        if self.mode == "normal":
            return total_q
        elif self.mode == "analysis":
            return K.concatenate([et1, et2, et3], axis=-1)

        def compute_output_shape(self, input_shape):
            if self.mode == "normal":
                return (input_shape[0], input_shape[1], 1)
            elif self.mode == "analysis":
                return (input_shape[0], input_shape[1], 3)

class LSTM_postprocess(Layer):
    def __init__(self, input_xd, hidden_size, seed=200,**kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTM_postprocess, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                 initializer=initializers.Orthogonal(seed=self.seed - 5),
                                 trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                       shape=(self.hidden_size, 4 * self.hidden_size),
                                       initializer=initializers.Orthogonal(seed=self.seed + 5),
                                       trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size, ),
                                    #initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTM_postprocess, self).build(input_shape)


    def reset_parameters(self):

        w_hh_data = K.eye(self.hidden_size)

        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

    def call(self, inputs_x):
        forcing = inputs_x  #[batch, seq_len, dim]

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  #[seq_len, batch, dim]

        seq_len = forcing_seqfir.shape[0]
        batch_size = forcing_seqfir.shape[1]
        h0 = K.zeros(shape= (batch_size, self.hidden_size))
        c0 = K.zeros(shape= (batch_size, self.hidden_size))
        h_x = (h0, c0)
        h_n, c_n = [], []
        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)

        for t in range(seq_len):
            h_0, c_0 = h_x

            gates =((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)

            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h,next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        return h_n, c_n











