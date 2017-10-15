# -*- coding: utf-8 -*- 

import numpy as np
import sys
import hook
from keras.models import Model, model_from_json
from keras.layers import Input, add, Masking, Activation, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, Flatten, Permute, RepeatVector, Permute, Dense, Lambda, Reshape
from keras.layers.merge import Concatenate, Dot, concatenate, multiply
from keras.backend import dropout, sum, sigmoid, binary_crossentropy, variable, random_uniform_variable, constant, int_shape, dot, is_keras_tensor
from keras.optimizers import Adam
from keras.initializers import Constant
from custom_layers.attentions import Attention
from custom_layers.averaging import Averaging
from custom_layers.features import Feature

def new_tensor_(name, shape):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    initial[0] = np.zeros(shape[1])

    return constant(initial, shape=shape, name=name)

def lambda_(x, r_dim, t_dim):
    W = new_tensor_('hier_W', (r_dim, t_dim))
    
    # Logit
    # is a function that maps probabilities ([0,1]) to R ([-inf, +inf])
    # L = ln((p/1-p)); p = 1/(1+e^-L)
    # Probability 0.5 correspond to a logit of 0. Negative logit correspond
    # to probabilities less than 0.5, positive > 0.5
    dot_ = dot(x, W)
    
    return sigmoid(dot_)

class KerasModel:
    def __init__(self, **kwargs):

        # **kwargs
        self.load_model = kwargs['load_model'] if 'load_model' in kwargs else None
        self.encoder = kwargs['encoder'] if 'encoder' in kwargs else None
        self.feature = kwargs['feature'] if 'feature' in kwargs else None

        # Hyperparams
        self.context_length = kwargs['context_length'] if 'context_length' in kwargs else 5
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        self.emb_dim = 300
        self.target_dim = 113
        self.dropout_ = 0.5
        self.learning_rate = 0.001
        self.feature_size = 600000
        self.lstm_dim = 100
        self.attention_dim = 100
        self.feature_dim = 50
        self.feature_input_dim = 70
        self.representation_dim = self.lstm_dim*2 + self.emb_dim if self.encoder != 'averaging' else self.emb_dim*3

        if self.feature:
            self.representation_dim += self.feature_dim

        # TODO: hook.acc_hook
        self.model_metrics = ['accuracy', 'mae']
        self.loss_f = 'binary_crossentropy'

        if self.load_model is not None:
            self.load_from_json_and_compile(self.load_model)

        else:

            # Use batch_shape(3D) when stateful=True
            # self.mention_representation = Input(batch_shape=(self.batch_size, self.emb_dim), name='input_3')
            # self.left_context = Input(batch_shape=(self.batch_size, self.context_length, self.emb_dim), name='input_1')
            # self.right_context = Input(batch_shape=(self.batch_size, self.context_length, self.emb_dim), name='input_2')
            # self.target = Input(batch_shape=(self.batch_size, self.target_dim))

            self.mention_representation = Input(shape=(self.emb_dim,), name='input_3')
            self.left_context = Input(shape=(self.context_length, self.emb_dim,), name='input_1')
            self.right_context = Input(shape=(self.context_length, self.emb_dim,), name='input_2')
            self.target = Input(shape=(self.target_dim,))

            # Dropout over mention_representation
            self.mention_representation_dropout = Dropout(self.dropout_)(self.mention_representation)

            # Context as list of Input tensors
            # self.context = [Input(batch_shape=(self.batch_size,self.emb_dim)) for i in range(self.context_length*2+1)]
            # self.left_context = self.context[:self.context_length]
            # self.right_context = self.context[self.context_length+1:]

            # LSTM
            if self.encoder == 'lstm':
                self.L_LSTM = LSTM(self.lstm_dim, recurrent_dropout=0.5, input_shape=int_shape(self.left_context))
                self.L_LSTM = self.L_LSTM(self.left_context)
                self.R_LSTM = LSTM(self.lstm_dim, recurrent_dropout=0.5, go_backwards=True)
                self.R_LSTM = self.R_LSTM(self.right_context)

                self.context_representation = concatenate([self.L_LSTM, self.R_LSTM], axis=1)

            # Averaging
            if self.encoder == 'averaging':
                self.context_representation = Averaging(concat_axis=1, sum_axis=1)
                self.context_representation = self.context_representation([self.left_context, self.right_context])

            # LSTM + Attentions
            if self.encoder == 'attentive':
                self.L_biLSTM = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, input_shape=int_shape(self.left_context)))
                self.L_biLSTM = self.L_biLSTM(self.left_context)
                self.R_biLSTM = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, input_shape=int_shape(self.left_context)))
                self.R_biLSTM = self.R_biLSTM(self.right_context)

                self.LR_biLSTM = add([self.L_biLSTM, self.R_biLSTM])

                # Attentive encoder
                attention_output = Attention()(self.LR_biLSTM)

                # self.context_representation = dot(self.LR_biLSTM, xxx)
                self.context_representation = attention_output

            # Logistic Regression
            if self.feature:
                self.feature_input = Input(shape=(self.feature_input_dim,), dtype='int32', name='input_4')
                self.feature_representation = Feature(F_emb_shape=(self.feature_size, self.feature_dim), F_emb_name='feat_emb', reduce_sum_axis=1, dropout=self.dropout_)
                self.feature_representation = self.feature_representation(self.feature_input)
                
                self.representation = concatenate([self.mention_representation_dropout, self.context_representation, self.feature_representation], axis=1) # is_keras_tensor=True
           
            else:
                self.representation = concatenate([self.mention_representation_dropout, self.context_representation], axis=1) # is_keras_tensor=True

            # TODO: Missing --hier part...
            # ...
            # self.W = new_tensor_('hier_W', (self.representation_dim, self.target_dim))

            # Used during prediction phase
            # TODO: try to use Dot() layer instead of Lambda
            # DON'T use Activation() layer!
            self.distribution_ = Lambda(lambda_, name='output_1')
            self.distribution_.arguments = {'r_dim': self.representation_dim, 't_dim': self.target_dim}
            self.distribution = self.distribution_(self.representation) # dot and sigmoid

            # Used during model compilation
            self.optimizer_adam = Adam(lr=self.learning_rate)

            # Prepare inputs list
            if self.feature:
                inputs = [self.left_context, self.right_context, self.mention_representation, self.feature_input]
            else:
                inputs = [self.left_context, self.right_context, self.mention_representation]

            # Creation and compilation
            self.model = Model(inputs=inputs, outputs=self.distribution)       
            self.model.compile(optimizer=self.optimizer_adam, metrics=self.model_metrics, loss=self.loss_f)


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

        print 'Loading model from JSON...'

        self.model = model_from_json(open(options['json_path']).read())
        self.model.compile(loss=options['loss'], optimizer=options['optimizer'], metrics=options['metrics'])
        self.model.load_weights(option['weights_path'])
        
        return self.model

    # steps_per_epoch = tot.samples / batch size
    def train_model(self, batcher, steps_per_epoch=1, epochs=1, shuffle=False, verbose=0):
        assert(batcher is not None)

        def _generate(batcher):
            while 1:
                context_data, mention_representation_data, target_data, feature_data = batcher.next()
                inputs = dict({
                        'input_1': context_data[:,:self.context_length,:], # TODO: external param
                        'input_2': context_data[:,self.context_length+1:,:],
                        'input_3': mention_representation_data
                    })

                if self.feature:
                    inputs['input_4'] = feature_data

                yield(inputs, {
                        'output_1': target_data
                    })

        if self.model is not None:
            return self.model.fit_generator(_generate(batcher), steps_per_epoch, epochs=epochs, shuffle=shuffle, verbose=verbose) # steps_per_epoch=2000