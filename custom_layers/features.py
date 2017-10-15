# -*- coding: utf-8 -*- 

import tensorflow as tf
from keras import backend as K
import keras.initializers as initializers
from keras.engine.topology import Layer

class Feature(Layer):
    def __init__(self, F_emb_shape=None, F_emb_name='', reduce_sum_axis=1, dropout=1, **kwargs):
        assert(F_emb_shape is not None)

        self.init = initializers.get('random_uniform')
        self.F_emb_shape = F_emb_shape
        self.F_emb_name = F_emb_name
        self.reduce_sum_axis = reduce_sum_axis
        self.dropout = dropout

        super(Feature, self).__init__(**kwargs)

    def build(self, input_shape):
        self.F = self.add_weight(self.F_emb_shape,
                                 initializer=self.init,
                                 name='{}_F'.format(self.F_emb_name))

        super(Feature, self).build(input_shape)

    def call(self, x):
        f_rep = tf.nn.embedding_lookup(self.F, x)
        f_rep = tf.reduce_sum(f_rep, self.reduce_sum_axis)
        self.f_rep_out = K.dropout(f_rep, self.dropout)

        return self.f_rep_out
    
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.f_rep_out)