# -*- coding: utf-8 -*- 

from keras import backend as K
from keras.engine.topology import Layer

class Averaging(Layer):
    def __init__(self, concat_axis=1, sum_axis=1, **kwargs):
        self.concat_axis = concat_axis
        self.sum_axis = sum_axis

        super(Averaging, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Averaging, self).build(input_shape)

    def call(self, inputs):
        l_context_sum = K.sum(inputs[0], axis=self.sum_axis, keepdims=False)
        r_context_sum = K.sum(inputs[1], axis=self.sum_axis, keepdims=False)
        self.concat_output = K.concatenate([l_context_sum, r_context_sum], axis=self.concat_axis)

        return self.concat_output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        return K.int_shape(self.concat_output)