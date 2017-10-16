# -*- coding: utf-8 -*- 

from keras import backend as K
from keras.engine.topology import Layer
from utils import create_prior, random_uniform_custom

class Hier(Layer):
    def __init__(self,process_hier=False, label2id_path=None, target_dim=None, V_emb_shape=None,
        V_emb_name='hier', return_logit=False, **kwargs):

        assert(V_emb_shape is not None)

        self.process_hier = process_hier
        self.V_emb_shape = V_emb_shape
        self.V_emb_name = V_emb_name
        self.return_logit = return_logit

        if self.process_hier:
            assert(label2id_path is not None)
            assert(target_dim is not None)
            assert(V_emb_shape is not None and isinstance(V_emb_shape, tuple))

            self.label2id_path = label2id_path
            self.target_dim = target_dim
            self.S = create_prior(self.label2id_path)

            assert(self.S.shape == (self.target_dim, self.target_dim))
        
        super(Hier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.V = self.add_weight(self.V_emb_shape,
                                 initializer=random_uniform_custom(self.V_emb_shape, -0.01, 0.01),
                                 name='{}_V'.format(self.V_emb_name))

        super(Hier, self).build(input_shape)

    def call(self, x):
        if self.process_hier:
            self.S = K.constant(self.S, dtype='float32')
            self.W = K.dot(self.S, self.V)
            self.W = K.transpose(self.W)
            self.logit = K.dot(x, self.W)

        else:
            self.logit = K.dot(x, self.V)

        self.distribution = K.sigmoid(self.logit)

        if self.return_logit:
            return [self.logit, self.distribution]
        return self.distribution

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_logit:
            return [K.int_shape(self.logit), K.int_shape(self.distribution)]
        return K.int_shape(self.distribution)