#-*- coding: utf-8 -*-
# import keras as K
import tensorflow as tf
import numpy as np
import sys
from keras.models import Model as KerasModel
from keras.layers import Input, add
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, Flatten, Permute, RepeatVector, Permute, Dense
from keras.layers.merge import Concatenate, Dot, concatenate, multiply
from keras.models import Sequential
from keras.backend import dropout, sigmoid, binary_crossentropy, variable, random_uniform_variable, constant
from keras.optimizers import Adam
from keras.initializers import Constant

class Model:
    def __init__(self, encoder='lstm', dev_test=False):
        assert(encoder in ['lstm', 'attentive'])

        # Input
        self.dev_test = dev_test
        self.encoder = encoder

        # Hyperparams
        self.context_length = 10
        self.batch_size = 1000
        self.step_per_epoch = 2000
        self.nb_epochs = 5
        self.emb_dim = 300
        self.target_dim = 113
        self.dropout_ = 0.5
        self.learning_rate = 0.001
        self.feature_size = 600000
        self.lstm_dim = 100
        self.attention_dim = 100 # dim of attention module
        self.feature_dim = 50 # dim of feature representation
        self.feature_input_dim = 70
        self.rep_dim = self.lstm_dim * 2 + self.emb_dim # if encoder is not 'averanging'
        # self.rep_dim += self.feature_dim # if --feature
        self.model_metrics = ['accuracy']

        # Not needed anymore! We're using Functional API
        # self.model = Sequential()

        # Placeholders with Tensorflow
        # self.keep_prob = K.backend.placeholder(dtype='float32') # K.backend.placeholder((2, 3), dtype='float32')
        # self.mention_representation = K.backend.placeholder((None, self.emb_dim), dtype='float32')
        # self.context = [K.backend.placeholder((None, self.emb_dim), dtype='float32') for _ in range((self.context_length * 2) + 1)]
        # self.target = K.backend.placeholder((None, self.target_dim), dtype='float32')

        # Dropout and split context into L/R
        # Dropout with Keras has a problem... so we have to use tf.nn.dropout!
        # mention_representation_dropout = tf.nn.dropout(mention_representation, keep_prob)

        # batch_shape needed when RNN is stateful
        # self.mention_representation = Input(shape=(self.emb_dim,))
        # self.left_context = Input(batch_shape=(self.batch_size,self.context_length,self.emb_dim))
        # self.right_context = Input(batch_shape=(self.batch_size,self.context_length,self.emb_dim))
        # self.target = Input(batch_shape=(self.batch_size,self.target_dim))

        self.mention_representation = Input(shape=(self.emb_dim,))
        self.left_context = Input(shape=(self.context_length,self.emb_dim,))
        self.right_context = Input(shape=(self.context_length,self.emb_dim,))
        self.target = Input(shape=(self.target_dim,))

        self.mention_representation_dropout = dropout(self.mention_representation, self.dropout_)
        # self.context = Input(batch_shape=(self.batch_size,self.context_length*2+1,self.emb_dim))
        # self.left_context = self.context[:self.context_length]
        # self.right_context = self.context[self.context_length + 1:]

        print self.left_context.shape, self.right_context.shape
        print 'Context placeholder created!'

        if self.encoder == 'lstm':
            self.L_LSTM, L_state1, L_state2 = LSTM(self.lstm_dim, return_sequences=True, return_state=True, input_shape=self.left_context.shape)(self.left_context)
            self.R_LSTM, R_state1, R_state2 = LSTM(self.lstm_dim, return_sequences=True, return_state=True, go_backwards=True)(self.right_context)
            self.context_representation = concatenate([L_state1, R_state1], axis=1)

            print self.L_LSTM
            print L_state1
            print L_state2
            print L_state1.shape
            print L_state2.shape
            print self.R_LSTM
            print R_state1
            print R_state2
            print R_state1.shape
            print R_state2.shape

        # if --attentive (LSTM + Attentions)
        if self.encoder == 'attentive':
            self.LF_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True, input_shape=self.left_context.shape)(self.left_context)
            self.LB_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True, go_backwards=True)(self.left_context)
            self.L_biLSTM = Concatenate([self.LF_oneLSTM, self.LB_oneLSTM])
            self.RF_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True)(self.right_context)
            self.RB_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True, go_backwards=True)(self.right_context)
            self.R_biLSTM = Concatenate([self.RF_oneLSTM, self.RB_oneLSTM])

            # self.left_biLSTM = Bidirectional(self.left_oneLSTM)
            # self.right_biLSTM = Bidirectional(self.right_oneLSTM)ù

            print 'biLSTM created!'

            # !Dev!
            if self.dev_test:
                print 'Got murdered :('
                sys.exit(1)

            self.LR_biLSTM = Concatenate([self.L_biLSTM, self.R_biLSTM])

            self.attention = Dense(self.attention_dim, activation='tanh', input_shape=(self.lstm_dim*2,))(self.LR_biLSTM)
            self.attention = Flatten()(self.attention)
            self.attention = Activation('softmax')(self.attention)
            self.attention = RepeatVector(self.lstm_dim)(self.attention)
            self.attention = Permute([2, 1])(self.attention)

            self.context_representation = merge([self.activations, self.attention], mode='mul')

        # Missing --feature part...
        # ...
        self.representation = concatenate([self.mention_representation_dropout, self.context_representation], axis=1)
        self.representation = Dense(500)(self.representation) # Don't know if it's correct

        # Missing --hier part...
        # ...
        self.W = self.create_weight_variable('hier_W', (self.rep_dim, self.target_dim))

        print self.W
        print self.representation

        # TODO: remove tf.matmul and seek for a Keras implementation
        self.logit = tf.matmul(self.representation, self.W)

        # Used during prediction phase
        self.distribution = sigmoid(self.logit)

        # Used during model compilation
        self.loss_f = tf.reduce_mean(binary_crossentropy(self.logit, self.target, from_logits=True))
        self.optimizer_adam = Adam(lr=self.learning_rate)

        # Creating model
        # 'Tensor' object has no attribute '_keras_history'

        print self.left_context._keras_history
        print self.right_context._keras_history
        print self.mention_representation._keras_history
        print self.representation._keras_history

        self.model = KerasModel(inputs=[self.left_context, self.right_context, self.mention_representation], outputs=[self.representation])

    def set_attention_layer(self, model):
        # Set Attentions...
        return True

    def build(self):
        if self.model is not Null:
            self.model.compile(optimizer=self.optimizer_adam, metrics=self.model_metrics, loss=self.loss_f, batch_size=self.batch_size)

    def get_model_summary(self):
        if self.model is not Null:
            return self.model.summary()

    def get_model(self):
        if self.model is not Null:
            return self.model

    def create_weight_variable(self, name, shape, pad=True):
        initial = np.random.uniform(-0.01, 0.01, size=shape)
        # initial = random_uniform_variable(shape=shape, -0.01, 0.01)

        if pad == True:
            initial[0] = np.zeros(shape[1])

        # TODO: insert Constant someway...
        # initial = variable(initial, name=name, dtype='float32')
        # initial = tf.constant_initializer(initial)
        initial = constant(initial, shape=shape, name=name)
        return initial
