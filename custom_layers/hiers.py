# -*- coding: utf-8 -*- 

from keras import backend as K
from keras.engine.topology import Layer
from utils import create_prior, random_uniform_custom

# Building a Keras Custom Layer
# https://keras.io/layers/writing-your-own-keras-layers/

# Hierarchical Label Encoding
class Hier(Layer):
    def __init__(self, process_hier=False, label2id_path=None, target_dim=None, V_emb_shape=None,
        V_emb_name='hier', return_logit=False, **kwargs):
    
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

            # Create a sparse binary matrix from the 'label2id' file 
            self.S = create_prior(self.label2id_path)

            assert(self.S.shape == (self.target_dim, self.target_dim))
        
        super(Hier, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a weight matrix V with random_uniform_custom initializer
        self.V = self.add_weight(self.V_emb_shape,
                                 initializer=random_uniform_custom(self.V_emb_shape, -0.01, 0.01),
                                 name='{}_V'.format(self.V_emb_name))

        super(Hier, self).build(input_shape)

    def call(self, x):
        if self.process_hier:

            # Make S matrix a constant sparse binary matrix
            # where each type is encoded inside each column
            self.S = K.constant(self.S, dtype='float32')
            
            # Calculate W and then its transpose
            self.W = K.dot(self.S, self.V)
            self.W = K.transpose(self.W)

            # [...]For Tensorflow: It's a name that it is thought to imply
            # that this Tensor is the quantity that is being mapped to probabilities by the Softmax
            # https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
            self.logit = K.dot(x, self.W)

        else:
            self.logit = K.dot(x, self.V)

        # Activation function to logit (that's the output!)
        self.distribution = K.sigmoid(self.logit)

        # Return logit too if required
        if self.return_logit:
            return [self.logit, self.distribution]
        return self.distribution

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_logit:
            return [K.int_shape(self.logit), K.int_shape(self.distribution)]
        return K.int_shape(self.distribution)