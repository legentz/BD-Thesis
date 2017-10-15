# -*- coding: utf-8 -*- 

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

# This solves issues of pop(-2) overflow in Tensorflow environment
def dot_(x, kernel):
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

class Attention(Layer):
    def __init__(self, W_regularizer=None, u_regularizer=None, W_constraint=None, u_constraint=None, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('random_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.u_constraint = constraints.get(u_constraint)

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        uit = dot_(x, self.W)
        uit = K.tanh(uit)

        ait = dot_(uit, self.u)
        a = K.exp(ait)

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]