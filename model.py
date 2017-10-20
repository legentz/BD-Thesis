# -*- coding: utf-8 -*- 

from __future__ import print_function

from shimaoka.hook import hook
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
import datetime, sys

class KerasModel:
    def __init__(self, hyper=None, **kwargs):

        print('--> Creating model')

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

    def compile_model(self):

        print('--> Compiling model')

        # Input tensors
        mention_representation = Input(shape=(self.emb_dim,), name='input_3')
        left_context = Input(shape=(self.context_length, self.emb_dim,), name='input_1')
        right_context = Input(shape=(self.context_length, self.emb_dim,), name='input_2')
        target = Input(shape=(self.target_dim,))

        if self.feature:
            feature_input = Input(shape=(self.feature_input_dim,), dtype='int32', name='input_4')

        # Dropout over mention_representation
        mention_representation_dropout = Dropout(self.dropout)
        mention_representation_dropout = mention_representation_dropout(mention_representation)

        # LSTM
        if self.encoder == 'lstm':
            context_representation = self.__lstm_encoder(left_context, right_context)

        # Averaging
        if self.encoder == 'averaging':
            context_representation = self.__averaging_encoder(left_context, right_context)

        # LSTM + Attentions
        if self.encoder == 'attentive':
            context_representation = self.__attentive_encoder(left_context, right_context)

        if self.feature:
            feature_representation = self.__process_features(feature_input)
            
            representation = concatenate(
                [mention_representation_dropout, context_representation, feature_representation],
                axis=1
            ) 
       
        else:
            representation = concatenate(
                [mention_representation_dropout, context_representation],
                axis=1
            )

        distribution = self.__process_hier(representation)

        # Prepare inputs/outputs list
        if self.feature:
            inputs = [left_context, right_context, mention_representation, feature_input]
        else:
            inputs = [left_context, right_context, mention_representation]

        outputs = [distribution]

        # Creation and compilation
        self.model = Model(inputs=inputs, outputs=outputs)       
        self.model.compile(optimizer=self.optimizer_adam, metrics=self.metrics, loss=self.loss)

    def __averaging_encoder(self, left_context, right_context):
        context_representation = Averaging(concat_axis=1, sum_axis=1)
        context_representation = context_representation([left_context, right_context])

        return context_representation

    def __lstm_encoder(self, left_context, right_context):
        L_LSTM = LSTM(self.lstm_dim, input_shape=int_shape(left_context))
        L_LSTM = L_LSTM(left_context)
        R_LSTM = LSTM(self.lstm_dim, go_backwards=True)
        R_LSTM = R_LSTM(right_context)

        context_representation = concatenate([L_LSTM, R_LSTM], axis=1)

        return context_representation

    def __attentive_encoder(self, left_context, right_context):
        L_biLSTM = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, input_shape=int_shape(left_context)), merge_mode='concat')
        L_biLSTM = L_biLSTM(left_context)
        R_biLSTM = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, input_shape=int_shape(left_context)), merge_mode='concat')
        R_biLSTM = R_biLSTM(right_context)

        # LR_biLSTM = add([L_biLSTM, R_biLSTM])
        # Using concatenate() on axis=1 because in NFGEC the list of tensors of Left and Right
        # are "concatenated" in an list of Tensors (20 elements).
        # In this case, (None, 10, 200) + (None, 10, 200) = (None, 20, 200)
        # where 20 is the timesteps or n. of words
        LR_biLSTM = concatenate([L_biLSTM, R_biLSTM], axis=1)

        context_representation = Attention(attention_hidden_dim=self.attention_dim)(LR_biLSTM)
        # context_representation = Attention()([L_biLSTM, R_biLSTM])

        return context_representation

    def __process_features(self, feature_input):
        feature_representation = Feature(
            F_emb_shape=(self.feature_size, self.feature_dim),
            F_emb_name='feat_emb',
            reduce_sum_axis=1,
            dropout=self.dropout
        )
        feature_representation = feature_representation(feature_input)

        return feature_representation

    def __process_hier(self, representation):
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

        return distribution

    def get_model_summary(self, print_fn=None):
        if self.model is not None:
            if print_fn is not None:
                return self.model.summary(print_fn=print_fn)
            return self.model.summary()

    def get_model(self):
        if self.model is not None:
            return self.model

    def set_model(self, new_model):
        if new_model is not None:
            self.model = new_model

    def set_model_weights(self, weights_path):

        print('--> Restoring model weights')

        if self.model is not None:
            self.model.load_weights(weights_path)

    def save_model(self, weights_path=None): # json_path=None

        print('--> Saving model')

        # Used to produce different backup .h5/.json
        now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')

        # Save
        # json = self.model.to_json()
        # json_path = json_path + now + '.json'
        weights_path = weights_path + now + '.h5'

        # open(json_path, 'w').write(json)
        self.model.save_weights(weights_path)

    # steps_per_epoch = tot.samples / batch size
    def train_model(self, batcher, steps_per_epoch=1, epochs=1, shuffle=False, verbose=0):

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

        return self.model.fit_generator(
            _generate(batcher),
            steps_per_epoch,
            callbacks=[on_begin_callback, on_end_callback],
            epochs=epochs,
            shuffle=shuffle,
            verbose=verbose
        )

    def predict_and_evaluate_model(self, batcher, batch_size=1, acc_hook=False, id2label=None,
        show_results_vector=False, save_as_txt=None, verbose=0):

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


