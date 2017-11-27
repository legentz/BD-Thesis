# -*- coding: utf-8 -*- 

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from utils import random_uniform_custom

# Building a Keras Custom Layer
# https://keras.io/layers/writing-your-own-keras-layers/

# AttentionWithContext
# https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

# This solves issues of pop(-2) overflow in Tensorflow environment
def dot_(x, kernel):
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

# Attentive Encoder
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        shape_W = (input_shape[-1], input_shape[-1],)
        shape_u = (input_shape[-1],)

        # Create a matrix W with the random_uniform_custom initializer 
        self.W = self.add_weight(shape_W,
                                 initializer=random_uniform_custom(shape_W, -0.01, 0.01),
                                 name='{}_W'.format(self.name))

        # Create a matrix u with the random_uniform_custom initializer 
        self.u = self.add_weight(shape_u,
                                 initializer=random_uniform_custom(shape_u, -0.01, 0.01),
                                 name='{}_u'.format(self.name))

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):

        # x will be the Tensor that represents the addition of both
        # left and right biLSTMs

        # Formula (4) and (5) from Shimaoka paper (https://arxiv.org/pdf/1606.01341v1.pdf)
        uit = dot_(x, self.W)
        uit = K.tanh(uit)
        ait = dot_(uit, self.u)
        a = K.exp(ait)

        # Making standard normalization
        # Formula (6) from Shimaoka paper (https://arxiv.org/pdf/1606.01341v1.pdf)
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        # Obtaining the context vector (v_c)
        # Formula (7) from the Shimaoka paper (https://arxiv.org/pdf/1606.01341v1.pdf) 
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]