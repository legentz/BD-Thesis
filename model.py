#-*- coding: utf-8 -*-
# import keras as K
import tensorflow as tf
import numpy as np
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, Flatten, Permute, RepeatVector, Permute, Dense
from keras.layers.merge import Concatenate, Dot
from keras.models import Sequential
from keras.backend import dropout, sigmoid, binary_crossentropy
from keras.optimizers. import Adam

class Model:
    def __init__(self, dev_test=False):
        self.dev_test = dev_test

        #Hyperparams
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
        self.rep_dim += self.feature_dim # if --feature
        self.model_metrics = ['accuracy']

        self.model = Sequential()

        # Placeholders with Tensorflow
        # self.keep_prob = K.backend.placeholder(dtype='float32') # K.backend.placeholder((2, 3), dtype='float32')
        # self.mention_representation = K.backend.placeholder((None, self.emb_dim), dtype='float32')
        # self.context = [K.backend.placeholder((None, self.emb_dim), dtype='float32') for _ in range((self.context_length * 2) + 1)]
        # self.target = K.backend.placeholder((None, self.target_dim), dtype='float32')

        # Trying to use ---- tensor
        self.mention_representation = Input(shape=(self.emb_dim,))
        self.context = Input(shape=(self.emb_dim,))
        self.target = Input(shape=(self.target_dim,))

        # Dropout and split context into L/R
        # Dropout with Keras has a problem... so we have to use tf.nn.dropout!
        # mention_representation_dropout = tf.nn.dropout(mention_representation, keep_prob)
        self.mention_representation_dropout = dropout(self.mention_representation, self.dropout_)
        self.left_context = self.context[:self.context_length]
        self.right_context = self.context[self.context_length + 1:]

        print 'Context placeholder created!'

        # if --attentive (LSTM + Attentions)
        self.left_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, return_state=True, stateful=True)(self.left_context) # stateful=True,
        self.right_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, return_state=True, stateful=True, go_backwards=True)(self.right_context) # stateful=True,
        self.left_biLSTM = Bidirectional(self.left_oneLSTM, merge_mode='concat') # (self.left_context)
        self.right_biLSTM = Bidirectional(self.right_oneLSTM, merge_mode='concat') # (self.right_context)

        print 'biLSTM created!'

        # Updating model
        self.model.add(self.left_biLSTM)
        self.model.add(Dropout(0.5))
        self.model.add(self.right_biLSTM)
        self.model.add(Dropout(0.5))

        print 'biLSTM added to model!'

        # Dev
        if self.dev_test:
            print 'Got murdered :('
            sys.exit(1)

        self.merge_biLSTM = merge([self.left_biLSTM, self.right_biLSTM], mode='sum')

        self.attention = Dense(self.attention_dim, activation='tanh', input_shape=self.lstm_dim*2)(self.merge_biLSTM)
        self.attention = Flatten()(self.attention)
        self.attention = Activation('softmax')(self.attention)
        self.attention = RepeatVector(self.lstm_dim)(self.attention)
        self.attention = Permute([2, 1])(self.attention)

        self.context_representation = merge([self.activations, self.attention], mode='mul')

        # Missing --feature part...
        # ...
        self.representation = Concatenate([self.mention_representation_dropout, self.context_representation], axis=1)

        # Missing --hier part...
        # ...
        self.W = self.create_weight_variable('hier_W', (self.rep_dim, self.target_dim))
        self.logit = Dot(self.representation, self.W)

        self.distribution = sigmoid(self.logit)

        self.loss_f = np.mean(binary_crossentropy(self.logit, self.target, from_logits=True))
        self.optimizer_adam = Adam(lr=self.learning_rate)

    def set_attention_layer(self, model):
        # Set Attentions...
        return True

    def compile_model(self):
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
        # initial = K.random_uniform_variable(shape=shape, -0.01, 0.01)

        if pad == True:
            initial[0] = np.zeros(shape[1])

        # initial = tf.constant_initializer(initial)
        initial = tf.contrib.keras.initializers.Constant(initial)

        return K.backend.variable(value=initial)
