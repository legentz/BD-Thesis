# -*- coding: utf-8 -*- 

from __future__ import print_function
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
from keras.callbacks import LambdaCallback
from custom_layers.attentions import Attention
from custom_layers.averaging import Averaging
from custom_layers.features import Feature
from custom_layers.hiers import Hier

class KerasModel:
    def __init__(self, hyper=None, **kwargs):
        assert(hyper is not None)

        print('--> Creating model')

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


        # # TODO: to be continued
        # # Load module from .json/.h5...
        # if self.load_model:
        #     # TODO: Fix load_model options
        #     self.load_from_json_and_compile(self.load_model)

        # # ...or create a new one
        # else:
    def compile_model(self):

        # Input tensors
        mention_representation = Input(shape=(self.emb_dim,), name='input_3')
        left_context = Input(shape=(self.context_length, self.emb_dim,), name='input_1')
        right_context = Input(shape=(self.context_length, self.emb_dim,), name='input_2')
        target = Input(shape=(self.target_dim,))

        # Dropout over mention_representation
        mention_representation_dropout = Dropout(self.dropout)(mention_representation)

        # LSTM
        if self.encoder == 'lstm':
            L_LSTM = LSTM(self.lstm_dim, recurrent_dropout=self.dropout, input_shape=int_shape(left_context))
            L_LSTM = L_LSTM(left_context)
            R_LSTM = LSTM(self.lstm_dim, recurrent_dropout=self.dropout, go_backwards=True)
            R_LSTM = R_LSTM(right_context)

            context_representation = concatenate([L_LSTM, R_LSTM], axis=1)

        # Averaging
        if self.encoder == 'averaging':
            context_representation = Averaging(concat_axis=1, sum_axis=1)
            context_representation = context_representation([left_context, right_context])

        # LSTM + Attentions
        if self.encoder == 'attentive':
            L_biLSTM = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, input_shape=int_shape(left_context)))
            L_biLSTM = L_biLSTM(left_context)
            R_biLSTM = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, input_shape=int_shape(left_context)))
            R_biLSTM = R_biLSTM(right_context)

            LR_biLSTM = add([L_biLSTM, R_biLSTM])

            # Attentive encoder
            context_representation = Attention()(LR_biLSTM)

        # Logistic Regression
        if self.feature:
            feature_input = Input(shape=(self.feature_input_dim,), dtype='int32', name='input_4')
            feature_representation = Feature(
                F_emb_shape=(self.feature_size, self.feature_dim),
                F_emb_name='feat_emb',
                reduce_sum_axis=1,
                dropout=self.dropout
            )
            feature_representation = feature_representation(feature_input)
            
            representation = concatenate([mention_representation_dropout, context_representation, feature_representation], axis=1) # is_keras_tensor=True
       
        else:
            representation = concatenate([mention_representation_dropout, context_representation], axis=1) # is_keras_tensor=True

        # Hier part
        if self.hier:
            V_emb_shape = (self.target_dim, self.representation_dim) 
        else:
            V_emb_shape = (self.representation_dim, self.target_dim)

        distribution = Hier(
            process_hier=self.hier,
            label2id_path=self.label2id_path,
            target_dim=self.target_dim,
            V_emb_shape=V_emb_shape,
            V_emb_name='hier',
            return_logit=False,
            name='output_1'
        )
        distribution = distribution(representation)

        # Prepare inputs/outputs list
        if self.feature:
            inputs = [left_context, right_context, mention_representation, feature_input]
        else:
            inputs = [left_context, right_context, mention_representation]

        outputs = [distribution]

        # Creation and compilation
        self.model = Model(inputs=inputs, outputs=outputs)       
        self.model.compile(optimizer=self.optimizer_adam, metrics=self.metrics, loss=self.loss)


    def get_model_summary(self):
        if self.model is not None:
            return self.model.summary()

    def get_model(self):
        if self.model is not None:
            return self.model

    def save_to_json(self, json_path=None, weights_path=None):
        assert(json_path is not None)
        assert(weights_path is not None)
        assert(self.model is not None)

        print('--> Saving model')

        # Used to produce different backup .h5/.json
        now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')

        # Save
        json = self.model.to_json()
        json_path = json_path + now + '.json'
        weights_path = weights_path + now + '.h5'

        open(json_path, 'w').write(json)
        self.model.save_weights(weights_path)

    def load_from_json_and_compile(self, options=None):
        assert(options is not None)
        assert(options['json_path'] is not None)
        assert(options['metrics'] is not None)
        assert(options['loss'] is not None)
        assert(options['optimizer'] is not None)
        assert(options['weights_path'] is not None)

        print('--> Loading model from JSON...')

        self.model = model_from_json(open(options['json_path']).read())
        self.model.compile(loss=options['loss'], optimizer=options['optimizer'], metrics=options['metrics'])
        self.model.load_weights(option['weights_path'])
        
        return self.model

    # steps_per_epoch = tot.samples / batch size
    def train_model(self, batcher, steps_per_epoch=1, epochs=1, shuffle=False, verbose=0):
        assert(self.model is not None)
        assert(batcher is not None)

        print('--> Training model')

        on_begin_callback = LambdaCallback(
            on_train_begin=lambda logs: print('Started at ', datetime.datetime.now().strftime('%d-%m-%Y_%H:%M'))
        )
        on_end_callback = LambdaCallback(
            on_train_end=lambda logs: print('Ended at ', datetime.datetime.now().strftime('%d-%m-%Y_%H:%M'))
        )

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

        return self.model.fit_generator(_generate(batcher), steps_per_epoch, callbacks=[on_begin_callback, on_end_callback], epochs=epochs, shuffle=shuffle, verbose=verbose)

    def get_predictions(self, batcher, batch_size=1, acc_hook=False, id2label=None, show_results_vector=False, save_as_txt=None, verbose=0):
        assert(self.model is not None)
        assert(batcher is not None)

        print('--> Getting predictions')

        context_data, mention_representation_data, target_data, feature_data = batcher.next()
        inputs = dict({
            'input_1': context_data[:,:self.context_length,:],
            'input_2': context_data[:,self.context_length+1:,:],
            'input_3': mention_representation_data
        })

        if self.feature:
            inputs['input_4'] = feature_data

        results = self.model.predict(inputs, batch_size=batch_size, verbose=verbose)

        if show_results_vector:
            print(results)

        if acc_hook:
            hook.acc_hook(results, target_data)
       
            if save_as_txt is not None:
                if isinstance(save_as_txt, bool):
                    save_as_txt = 'NO_NAME'

                now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')
                hook.save_predictions(results, target_data, id2label, save_as_txt + now + '.txt')


