# -*- coding: utf-8 -*- 

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from utils import random_uniform_custom

# This solves issues of pop(-2) overflow in Tensorflow environment
def dot_(x, kernel):
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

class Attention(Layer):
    def __init__(self, attention_hidden_dim=None, **kwargs):
        self.attention_hidden_dim = attention_hidden_dim

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.attention_hidden_dim is None:
            self.attention_hidden_dim = input_shape[-1]

        shape_W = (input_shape[-1], self.attention_hidden_dim)
        shape_u = (self.attention_hidden_dim,)

        self.W = self.add_weight(shape_W,
                                 initializer=random_uniform_custom(shape_W, -0.01, 0.01),
                                 name='{}_W'.format(self.name))

        self.u = self.add_weight(shape_u,
                                 initializer=random_uniform_custom(shape_u, -0.01, 0.01),
                                 name='{}_u'.format(self.name))

        super(Attention, self).build(input_shape)

    def call(self, x):
        uit = K.dot(x, self.W)
        uit = K.tanh(uit)

        ait = dot_(uit, self.u)
        a = K.exp(ait)

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        a = K.softmax(a)

        weighted_input = x * a
        weighted_input = K.sum(weighted_input, axis=1)
        self.output_ = weighted_input

        return self.output_

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        return K.int_shape(self.output_)