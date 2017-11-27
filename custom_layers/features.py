# -*- coding: utf-8 -*- 

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from utils import random_uniform_custom

# Building a Keras Custom Layer
# https://keras.io/layers/writing-your-own-keras-layers/

# Hybrid Model (features learnt from data + hand-crafted features)
class Feature(Layer):
    def __init__(self, F_emb_shape=None, F_emb_name='', reduce_sum_axis=1, dropout=1, **kwargs):
        self.F_emb_shape = F_emb_shape
        self.F_emb_name = F_emb_name
        self.reduce_sum_axis = reduce_sum_axis
        self.dropout = dropout

        super(Feature, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a matrix F with the random_uniform_custom initializer
        self.F = self.add_weight(self.F_emb_shape,
                                 initializer=random_uniform_custom(self.F_emb_shape, -0.01, 0.01),
                                 name='{}_F'.format(self.F_emb_name))

        super(Feature, self).build(input_shape)

    def call(self, x):
        # embedding_lookup function retrieves rows of the params (x) tensor
        # https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
        f_rep = tf.nn.embedding_lookup(self.F, x)

        # Computes the sum of elements across dimensions (reduce_sum_axis) of a tensor
        # https://www.tensorflow.org/api_docs/python/tf/reduce_sum
        f_rep = tf.reduce_sum(f_rep, self.reduce_sum_axis)

        # Apply Dropout with 0.5 default value
        def dropped_inputs():
            return K.dropout(f_rep, self.dropout)

        # Use Dropout only during training phase
        self.f_rep_out = K.in_train_phase(dropped_inputs, f_rep)

        return self.f_rep_out
    
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.f_rep_out)