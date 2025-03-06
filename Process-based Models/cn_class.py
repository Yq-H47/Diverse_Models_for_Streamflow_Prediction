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

class local_EXPHYDRO(Layer):

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(local_EXPHYDRO, self).__init__(**kwargs)

# Define parameters
    def build(self, input_shape):
        self.f = self.add_weight(name='f', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)

        super(local_EXPHYDRO, self).build(input_shape)

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
        tmax = tmax * 3
        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))
        return melt
#Evapotranspiration calculation
    def soilbucket(self, s1, pet, f, smax, qmax):
        f = f / 10
        smax = smax * 1500
        qmax = qmax * 50
        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)
        return [et, qsub, qsurf]


    def step_do(self, step_in, states):
        s0 = states[0][:, 0:1]
        s1 = states[0][:, 1:2]

        p = step_in[:, 2:3]
        t = step_in[:, 1:2]
        pet = step_in[:, 0:1]

# Partition precipitation into rain and snow
        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)

        _m = self.snowbucket(s0, t, self.ddf, self.tmax)

        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):

        pet = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        prcp = inputs[:, :, 2:3]

        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        if self.mode == "normal":
            return qsub+qsurf
        elif self.mode == "analysis":
            return K.concatenate([s0, s1, m, et, qsub, qsurf], axis=-1)
#Select output variable
    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 6)

class local_Xinanjiang(Layer):
    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(local_Xinanjiang, self).__init__(**kwargs)
#Define parameters
    def build(self, input_shape):

        self.wum = self.add_weight(name='wum', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                   trainable=True)
        self.wlm = self.add_weight(name='wlm', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                   trainable=True)
        self.wdm = self.add_weight(name='wdm', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                   trainable=True)
        self.c = self.add_weight(name='c', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                 trainable=True)
        self.k1 = self.add_weight(name='k1', shape=(1,),
                                  initializer=initializers.Constant(value=0.5),
                                  constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                  trainable=True)
        self.k2 = self.add_weight(name='k2', shape=(1,),
                                  initializer=initializers.Constant(value=0.5),
                                  constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                  trainable=True)
        self.k3 = self.add_weight(name='k3', shape=(1,),
                                  initializer=initializers.Constant(value=0.5),
                                  constraint=constraints.min_max_norm(min_value=0, max_value=1, rate=0.9),
                                  trainable=True)
        super(local_Xinanjiang, self).build(input_shape)

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

        [_et1, _et2, _et3] = self.evapotranspiration(wu, wl, wd, pet)
        _dwu = p - _et1
        _dwl = p - _et1 - _et2
        _dwd = p - _et1 - _et2 - _et3
        wu_next = wu + K.clip(_dwu, -1e5, 1e5)
        wl_next = wl + K.clip(_dwl, -1e5, 1e5)
        wd_next = wd + K.clip(_dwd, -1e5, 1e5)
        #total_et = et1 + et2 + et3

        _runoff = self.runoff_production(wu_next, wl_next, wd_next, p, self.wum,self.wlu,self.wdm,self.b,self.c)

        [_surface_runoff, _interflow, _baseflow] = self.water_source_partition(_runoff,self.k1,self.k2,self.k3)

        step_out = [K.concatenate([wu_next, wl_next, wd_next], axis=1)]

        return step_out, [step_out]

    def call(self, inputs):

        pet = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        p = inputs[:, :, 2:3]
        new_inputs = K.concatenate((pet, p, tmean), axis=-1)

        init_states = [K.zeros((K.shape(new_inputs)[0], 3))]

        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)
        wu = outputs[:, :, 0:1]
        wl = outputs[:, :, 1:2]
        wd = outputs[:, :, 2:3]
        [et1, et2, et3] = self.evapotranspiration(wu, wl, wd, pet)
        runoff = self.runoff_production(wu,wd,wl,p,self.wum,self.wdm,self.wlm,self.b,self.c)
        [surface_runoff,interflow,baseflow]=self.water_source_partition(runoff, self.k1,self.k2,self.k3)
        total_q = self.flow_routing(surface_runoff, interflow, baseflow)

        if self.mode == "normal":
            return total_q
        elif self.mode == "analysis":
            return K.concatenate([et1, et2, et3], axis=-1)
#Select output variable
    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return input_shape[0], input_shape[1], 1
        elif self.mode == "analysis":
            return input_shape[0], input_shape[1], 3












