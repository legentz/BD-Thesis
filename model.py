#-*- coding: utf-8 -*-

import numpy as np
import sys
import hook
from keras.models import Model, model_from_json
from keras.layers import Input, add, Masking, Activation
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout, Flatten, Permute, RepeatVector, Permute, Dense, Lambda
from keras.layers.merge import Concatenate, Dot, concatenate, multiply
from keras.backend import dropout, sigmoid, binary_crossentropy, variable, random_uniform_variable, constant, int_shape, dot, is_keras_tensor
from keras.optimizers import Adam
from keras.initializers import Constant

def new_tensor_(name, shape, pad=True):
    initial = np.random.uniform(-0.01, 0.01, size=shape)

    if pad == True:
        initial[0] = np.zeros(shape[1])

    # TODO: insert Constant someway...
    # initial = variable(initial, name=name, dtype='float32')
    # initial = tf.constant_initializer(initial)
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

# Note on using statefulness in RNNs
# You can set RNN layers to be 'stateful', which means that the states
# computed for the samples in one batch will be reused as initial states
# for the samples in the next batch. This assumes a one-to-one mapping
# between samples in different successive batches.

# To enable statefulness:
#     - specify `stateful=True` in the layer constructor.
#     - specify a fixed batch size for your model, by passing
#         if sequential model:
#           `batch_input_shape=(...)` to the first layer in your model.
#         else for functional model with 1 or more Input layers:
#           `batch_shape=(...)` to all the first layers in your model.
#         This is the expected shape of your inputs
#         *including the batch size*.
#         It should be a tuple of integers, e.g. `(32, 10, 100)`.
#     - specify `shuffle=False` when calling fit().

# To reset the states of your model, call `.reset_states()` on either
# a specific layer, or on your entire model.

# Note on specifying the initial state of RNNs
# You can specify the initial state of RNN layers symbolically by
# calling them with the keyword argument `initial_state`. The value of
# `initial_state` should be a tensor or list of tensors representing
# the initial state of the RNN layer.

# You can specify the initial state of RNN layers numerically by
# calling `reset_states` with the keyword argument `states`. The value of
# `states` should be a numpy array or list of numpy arrays representing
# the initial state of the RNN layer.

class KerasModel:
    def __init__(self, **kwargs):

        # Class input
        self.load_model = kwargs['load_model'] if 'load_model' in kwargs else None
        self.encoder = kwargs['encoder'] if 'encoder' in kwargs else None

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

        # if encoder is not 'averanging'
        self.representation_dim = self.lstm_dim * 2 + self.emb_dim

        # if --feature
        # self.representation_dim += self.feature_dim

        # TODO: hook.acc_hook
        self.model_metrics = ['accuracy', 'mae']
        self.loss_f = 'binary_crossentropy'

        if self.load_model is not None:
            self.load_from_json_and_compile(self.load_model)

        else:

            # Use batch_shape(3D) when stateful=True
            self.mention_representation = Input(batch_shape=(self.batch_size, self.emb_dim), name='input_3')

            self.left_context = Input(batch_shape=(self.batch_size, self.context_length, self.emb_dim), name='input_1')
            self.right_context = Input(batch_shape=(self.batch_size, self.context_length, self.emb_dim), name='input_2')
            self.target = Input(batch_shape=(self.batch_size, self.target_dim))
            # self.context = [Input(batch_shape=(self.batch_size,self.emb_dim)) for i in range(self.context_length*2+1)]
            # self.left_context = self.context[:self.context_length]
            # self.right_context = self.context[self.context_length+1:]

            print 'left_context: ', int_shape(self.left_context)
            print 'right_context: ', int_shape(self.right_context)
            print 'mention_representation: ', int_shape(self.mention_representation)

            # LSTM
            if self.encoder == 'lstm':
                self.L_LSTM, l_1, l_2 = LSTM(self.lstm_dim, return_state=True, stateful=True, input_shape=int_shape(self.left_context))(self.left_context)
                self.R_LSTM, r_1, r_2 = LSTM(self.lstm_dim, return_state=True, stateful=True, go_backwards=True)(self.right_context)

                # self.context_representation = concatenate([L_state1, R_state1], axis=1)
                self.context_representation = concatenate([self.L_LSTM, self.R_LSTM], axis=1)
                # self.L_LSTM = LSTM(self.lstm_dim, stateful=True, return_sequences=True, input_shape=int_shape(self.left_context[0]), initial_state=self.left_context) # (self.left_context)
                # self.R_LSTM = LSTM(self.lstm_dim, stateful=True, return_sequences=True, go_backwards=True, initial_state=self.right_context)  # (self.right_context)
                # self.L_RNN, _ = RNN(self.L_LSTM, self.left_context)(self.left_context)
                # self.R_RNN, _ = RNN(self.R_LSTM, self.right_context)(self.right_context)
                # self.context_representation = concatenate([L_LSTM, R_LSTM], axis=1)

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
            self.mention_representation_dropout = Dropout(self.dropout_)(self.mention_representation)
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

            # Creation and compilation
            self.model = Model(inputs=[self.left_context, self.right_context, self.mention_representation], outputs=self.distribution)       
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

                yield({
                        'input_1': context_data[:,:self.context_length,:], # TODO: external param
                        'input_2': context_data[:,self.context_length+1:,:],
                        'input_3': mention_representation_data
                    }, {
                        'output_1': target_data
                    })

        if self.model is not None:
            return self.model.fit_generator(_generate(batcher), steps_per_epoch, epochs=epochs, shuffle=shuffle, verbose=verbose) # steps_per_epoch=2000