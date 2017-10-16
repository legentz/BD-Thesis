# -*- coding: utf-8 -*- 

import hook
import datetime
from keras.models import Model, model_from_json
from keras.layers import Input, add
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.backend import int_shape
from keras.optimizers import Adam
from custom_layers.attentions import Attention
from custom_layers.averaging import Averaging
from custom_layers.features import Feature
from custom_layers.hiers import Hier

class KerasModel:
    def __init__(self, hyper=None, **kwargs):
        assert(hyper is not None)

        print '--> Creating model'

        # **kwargs
        self.load_model = kwargs['load_model'] if 'load_model' in kwargs else False # TODO: put inside AIO config

        # Hyperparams
        self.encoder = hyper['encoder']
        self.context_length = hyper['context_length']
        self.batch_size = hyper['batch_size']
        self.dropout = hyper['dropout']
        self.learning_rate = hyper['learning_rate']
        self.emb_dim = hyper['emb_dim']
        self.target_dim = hyper['target_dim']

        # LSTM units
        self.lstm_dim = hyper['lstm_dim']

        # Attentive encoder units
        self.attention_dim = hyper['attention_dim']

        # Feature
        self.feature = hyper['feature']['process']
        self.feature_dim = hyper['feature']['dim']
        self.feature_input_dim = hyper['feature']['input_dim']
        self.feature_size = hyper['feature']['size']

        # Hier
        self.hier = hyper['hier']['process']
        self.label2id_path = hyper['hier']['label2id_path']

        # TODO: hook.acc_hook
        # Metrics and loss
        self.metrics = hyper['metrics']
        self.loss = hyper['loss']
        self.optimizer_adam = Adam(lr=self.learning_rate)

        # Representation
        if self.encoder != 'averaging':
            self.representation_dim = self.lstm_dim*2 + self.emb_dim
        else:
            self.representation_dim = self.emb_dim*3

        if self.feature:
            self.representation_dim += self.feature_dim


        # TODO: to be continued
        # Load module from .json/.h5...
        if self.load_model:
            # TODO: Fix load_model options
            self.load_from_json_and_compile(self.load_model)

        # ...or create a new one
        else:
            self.mention_representation = Input(shape=(self.emb_dim,), name='input_3')
            self.left_context = Input(shape=(self.context_length, self.emb_dim,), name='input_1')
            self.right_context = Input(shape=(self.context_length, self.emb_dim,), name='input_2')
            self.target = Input(shape=(self.target_dim,))

            # Dropout over mention_representation
            self.mention_representation_dropout = Dropout(self.dropout)(self.mention_representation)

            # LSTM
            if self.encoder == 'lstm':
                self.L_LSTM = LSTM(self.lstm_dim, recurrent_dropout=self.dropout, input_shape=int_shape(self.left_context))
                self.L_LSTM = self.L_LSTM(self.left_context)
                self.R_LSTM = LSTM(self.lstm_dim, recurrent_dropout=self.dropout, go_backwards=True)
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
                self.context_representation = Attention()(self.LR_biLSTM)

            # Logistic Regression
            if self.feature:
                self.feature_input = Input(shape=(self.feature_input_dim,), dtype='int32', name='input_4')
                self.feature_representation = Feature(F_emb_shape=(self.feature_size, self.feature_dim), F_emb_name='feat_emb', reduce_sum_axis=1, dropout=self.dropout)
                self.feature_representation = self.feature_representation(self.feature_input)
                
                self.representation = concatenate([self.mention_representation_dropout, self.context_representation, self.feature_representation], axis=1) # is_keras_tensor=True
           
            else:
                self.representation = concatenate([self.mention_representation_dropout, self.context_representation], axis=1) # is_keras_tensor=True

            # Hier part
            if self.hier:
                V_emb_shape = (self.target_dim, self.representation_dim) 
            else:
                V_emb_shape = (self.representation_dim, self.target_dim)

            self.distribution = Hier(
                process_hier=self.hier,
                label2id_path=self.label2id_path,
                target_dim=self.target_dim,
                V_emb_shape=V_emb_shape,
                V_emb_name='hier',
                return_logit=False,
                name='output_1'
            )
            self.distribution = self.distribution(self.representation)

            # Prepare inputs list
            if self.feature:
                inputs = [self.left_context, self.right_context, self.mention_representation, self.feature_input]
            else:
                inputs = [self.left_context, self.right_context, self.mention_representation]

            # Creation and compilation
            self.model = Model(inputs=inputs, outputs=self.distribution)       
            self.model.compile(optimizer=self.optimizer_adam, metrics=self.metrics, loss=self.loss)


    def get_model_summary(self):
        if self.model is not None:
            return self.model.summary()

    def get_model(self):
        if self.model is not None:
            return self.model

    def save_to_json(self, options=None):
        assert(options['json_path'] is not None)
        assert(options['weights_path'] is not None)

        print '--> Saving model'

        if self.model is not None:
            # Used to produce different backup .h5/.json
            now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')

            # Save
            json = self.model.to_json()
            json_path = options['json_path'] + now + '.json'
            weights_path = options['weights_path'] + now + '.h5'

            open(json_path, 'w').write(json)
            self.model.save_weights(weights_path)

    def load_from_json_and_compile(self, options=None):
        assert(options is not None)
        assert(options['json_path'] is not None)
        assert(options['metrics'] is not None)
        assert(options['loss'] is not None)
        assert(options['optimizer'] is not None)
        assert(options['weights_path'] is not None)

        print '--> Loading model from JSON...'

        self.model = model_from_json(open(options['json_path']).read())
        self.model.compile(loss=options['loss'], optimizer=options['optimizer'], metrics=options['metrics'])
        self.model.load_weights(option['weights_path'])
        
        return self.model

    # steps_per_epoch = tot.samples / batch size
    def train_model(self, batcher, steps_per_epoch=1, epochs=1, shuffle=False, verbose=0):
        assert(batcher is not None)

        print '--> Training model'

        def _generate(batcher):
            while 1:
                context_data, mention_representation_data, target_data, feature_data = batcher.next()
                inputs = dict({
                        'input_1': context_data[:,:self.context_length,:],
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