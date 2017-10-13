#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, add, Masking, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, Flatten, Permute, RepeatVector, Permute, Dense, Lambda
from keras.layers.merge import Concatenate, Dot, concatenate, multiply
from keras.models import Sequential
from keras.backend import dropout, sigmoid, binary_crossentropy, variable, random_uniform_variable, constant, int_shape, dot, is_keras_tensor
from keras.optimizers import Adam
from keras.initializers import Constant

def create_weight_variable(name, shape, pad=True):
    initial = np.random.uniform(-0.01, 0.01, size=shape)

    if pad == True:
        initial[0] = np.zeros(shape[1])

    # TODO: insert Constant someway...
    # initial = variable(initial, name=name, dtype='float32')
    # initial = tf.constant_initializer(initial)
    return constant(initial, shape=shape, name=name)

def lambda_(x, r_dim, t_dim):
    W = create_weight_variable('hier_W', (r_dim, t_dim))
    dot_ = dot(x, W)
    
    return sigmoid(dot_)

# Keras model wrapper
class KerasModel:
    # def __init__(self, load_model=options, encoder='lstm', compile_model=True):
    def __init__(self, **kwargs):
        # Class input
        self.load_model = kwargs['load_model'] if 'load_model' in kwargs else None
        self.encoder = kwargs['encoder'] if 'encoder' in kwargs else None

        # assert(encoder in ['lstm', 'attentive'])
        # self.compile_model = compile_model

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
        self.representation_dim = self.lstm_dim * 2 + self.emb_dim # if encoder is not 'averanging'
        # self.representation_dim += self.feature_dim # if --feature
        self.model_metrics = ['accuracy']
        self.loss_f = 'binary_crossentropy'

        # Loading and returnin model
        if self.load_model is not None:
            print 'Loading model from JSON...'
            self.load_from_json_and_compile(self.load_model)

        else:

            # TODO: check for the right shape(s)
            self.mention_representation = Input(shape=(self.emb_dim,), name='input_3')
            self.left_context = Input(shape=(self.context_length,self.emb_dim,), name='input_2')
            self.right_context = Input(shape=(self.context_length,self.emb_dim,), name='input_1')
            self.target = Input(shape=(self.target_dim,))

            # TODO: check this one if it's needed or not
            # self.mention_representation_dropout = dropout(self.mention_representation, self.dropout_)
            # self.context = Input(batch_shape=(self.batch_size,self.context_length*2+1,self.emb_dim))
            # self.left_context = self.context[:self.context_length]
            # self.right_context = self.context[self.context_length + 1:]

            print 'left_context: ', int_shape(self.left_context),
            print 'right_context: ', int_shape(self.right_context),
            print 'mention_representation: ', int_shape(self.mention_representation)

            # LSTM
            if self.encoder == 'lstm':
                self.L_LSTM, L_state1, L_state2 = LSTM(self.lstm_dim, return_sequences=True, return_state=True, input_shape=int_shape(self.left_context))(self.left_context)
                self.R_LSTM, R_state1, R_state2 = LSTM(self.lstm_dim, return_sequences=True, return_state=True, go_backwards=True)(self.right_context)
                self.context_representation = concatenate([L_state1, R_state1], axis=1)

            # LSTM + Attentions
            if self.encoder == 'attentive':
                self.LF_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True, input_shape=self.left_context.shape)(self.left_context)
                self.LB_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True, go_backwards=True)(self.left_context)
                self.L_biLSTM = Concatenate([self.LF_oneLSTM, self.LB_oneLSTM])
                self.RF_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True)(self.right_context)
                self.RB_oneLSTM = LSTM(self.lstm_dim, return_sequences=True, stateful=True, go_backwards=True)(self.right_context)
                self.R_biLSTM = Concatenate([self.RF_oneLSTM, self.RB_oneLSTM])

                self.LR_biLSTM = Concatenate([self.L_biLSTM, self.R_biLSTM])

                self.attention = Dense(self.attention_dim, activation='tanh', input_shape=(self.lstm_dim*2,))(self.LR_biLSTM)
                self.attention = Flatten()(self.attention)
                self.attention = Activation('softmax')(self.attention)
                self.attention = RepeatVector(self.lstm_dim)(self.attention)
                self.attention = Permute([2, 1])(self.attention)

                self.context_representation = merge([self.activations, self.attention], mode='mul')

            # TODO: Missing --feature part...
            # ...
            self.representation = concatenate([self.mention_representation, self.context_representation], axis=1) # is_keras_tensor=True
            # self.representation = Dense(500)(self.representation) # TODO; Don't know if it's needed

            # TODO: Missing --hier part...
            # ...
            self.W = create_weight_variable('hier_W', (self.representation_dim, self.target_dim))
            # self.W_ = Dense(113, input_shape=(self.representation_dim, self.target_dim))(self.W)
            # self.W = Dense(113, kernel_initializer='random_uniform', input_shape=(self.representation_dim, self.target_dim)) # TODO: need to pad [0]

            # self.logit = tf.matmul(self.representation, self.W)
            # self.logit = dot(self.representation, self.W)
            # self.logit_lambda = Lambda(self.lambda_)(self.representation)
            # self.logit = Dot(self.representation, self.W) TODO: try to use this instead of Lambda
            self.distribution_ = Lambda(lambda_, name='output_1')
            self.distribution_.arguments = {'r_dim': self.representation_dim, 't_dim': self.target_dim}
            self.distribution = self.distribution_(self.representation) # dot and sigmoid

            # Used during prediction phase
            # self.distribution = sigmoid(self.logit)
            # self.distribution = Dense(self.target_dim, activation='sigmoid')(self.logit_lambda)
            # self.distribution = Activation('sigmoid')(self.logit_lambda)

            print 'Is representation a Keras Tensor? ', is_keras_tensor(self.representation)
            print 'Is W a Keras Tensor? ', is_keras_tensor(self.W)
            # print 'Is logit_lambda a Keras Tensor? ', is_keras_tensor(self.logit_lambda)
            print 'Is distribution a Keras Tensor? ', is_keras_tensor(self.distribution)

            # Used during model compilation
            # self.loss_f = tf.reduce_mean(binary_crossentropy(self.logit, self.target, from_logits=True))
            self.optimizer_adam = Adam(lr=self.learning_rate)

            # Creating model...
            self.model = Model(inputs=[self.left_context, self.right_context, self.mention_representation], outputs=self.distribution)

            # Compiling model...
            print 'Compiling model...'
            
            self.model.compile(optimizer=self.optimizer_adam, metrics=self.model_metrics, loss=self.loss_f)

    # def set_attention_layer(self, model):
    #     # Set Attentions...
    #     return True

    def get_model_summary(self):
        if self.model is not None:
            return self.model.summary()

    def get_model(self):
        if self.model is not None:
            return self.model

    def save_to_json(self, options=None):
        assert(options['json_path'] is not None)
        assert(options['weights_path'] is not None)

        if self.model is not None:
            json = self.model.to_json()
            
            open(options['json_path'], 'w').write(json)

            self.model.save_weights(options['weights_path'])

    def load_from_json_and_compile(self, options=None):
        assert(options is not None)
        assert(options['json_path'] is not None)
        assert(options['metrics'] is not None)
        assert(options['loss'] is not None)
        assert(options['optimizer'] is not None)
        assert(options['weights_path'] is not None)

        self.model = model_from_json(open(options['json_path']).read())
        self.model.compile(loss=options['loss'], optimizer=options['optimizer'], metrics=options['metrics'])
        self.model.load_weights(option['weights_path'])
        
        return self.model

    def train_model(self, batcher, steps_per_epoch=1, epochs=1, shuffle=False, verbose=0):
        assert(batcher is not None)

        def _generate(batcher):
            while 1:
                context_data, mention_representation_data, target_data, feature_data = batcher.next()

                yield({
                        'input_1': context_data[:,:self.context_length,:],
                        'input_2': context_data[:,self.context_length+1:,:],
                        'input_3': mention_representation_data
                    }, {
                        'output_1': target_data
                    })

        if self.model is not None:
            return self.model.fit_generator(_generate(batcher), steps_per_epoch, epochs=epochs, shuffle=shuffle, verbose=verbose) # steps_per_epoch=2000