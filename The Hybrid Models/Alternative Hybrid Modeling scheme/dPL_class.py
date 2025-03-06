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











