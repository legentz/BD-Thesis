#-*- coding: utf-8 -*-

import keras as K
import tensorflow as tf
from sklearn.externals import joblib
import sys

# ... ?
def weight_variable(name, shape, pad=True):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    if pad == True:
        initial[0] = np.zeros(shape[1])
    initial = tf.constant_initializer(initial)
    return tf.get_variable(name=name, shape=shape, initializer=initial)

# def attentive_sum(inputs,input_dim, hidden_dim):
#     with tf.variable_scope("attention"):
#         seq_length = len(inputs)
#         W =  weight_variable('att_W', (input_dim,hidden_dim))
#         U =  weight_variable('att_U', (hidden_dim,1))
#         tf.get_variable_scope().reuse_variables()
#         temp1 = [tf.nn.tanh(tf.matmul(inputs[i],W)) for i in range(seq_length)]
#         temp2 = [tf.matmul(temp1[i],U) for i in range(seq_length)]
#         pre_activations = tf.concat(1,temp2)
#         attentions = tf.split(1, seq_length, tf.nn.softmax(pre_activations))
#         weighted_inputs = [tf.mul(inputs[i],attentions[i]) for i in range(seq_length)]
#         output = tf.add_n(weighted_inputs)
#     return output, attentions

##
## Load dicts and datasets
##
dicts = joblib.load("data/Wiki/dicts_figer.pkl")

train_dataset = joblib.load("data/Wiki/train_figer.pkl")
dev_dataset = joblib.load("data/Wiki/dev_figer.pkl")
test_dataset = joblib.load("data/Wiki/test_figer.pkl")

print "train_dataset_size", train_dataset["data"].shape[0]
print "dev_dataset_size", dev_dataset["data"].shape[0]
print "test_dataset_size", test_dataset["data"].shape[0]

##
## Hyperparams
##
context_length = 10
batch_size = 1000
step_per_epoch = 2000
nb_epochs = 5
emb_dim = 300
target_dim = 113
dropout_ = 0.5
learning_rate = 0.001
feature_size = 600000
lstm_dim = 100
attention_dim = 100 # dim of attention module 
feature_dim = 50 # dim of feature representation
feature_input_dim = 70
rep_dim = lstm_dim * 2 + emb_dim # if encoder is not 'averanging'
rep_dim += feature_dim # if --feature
# LSTM_activation = 'tanh'
# LSTM_stateful = True
# LSTM_return_seq = True
model_metrics = ['accuracy']
# loss_f = 'categorical_crossentropy'
# merge_biLSTM = 'concat'

##
## Model
##

# Placeholders with Tensorflow
keep_prob = K.backend.placeholder(dtype='float32') # K.backend.placeholder((2, 3), dtype='float32')
mention_representation = K.backend.placeholder((None, emb_dim), dtype='float32')
context = [K.backend.placeholder((None, emb_dim), dtype='float32') for _ in range((context_length * 2) + 1)]
target = K.backend.placeholder((None, target_dim), dtype='float32')

model = K.models.Sequential()

# Embedding layer
# ...
embedding_layer = K.layers.Embedding(len(dicts['word2id'].keys()) + 1,
                            len(dicts['id2vec']),
                            weights=[dicts['id2vec']],
                            input_length=emb_dim,
                            trainable=False)

print 'Embedded!'

# Dropout and split context into L/R
# Dropout with Keras has a problem... so we have to use tf.nn.dropout!
# mention_representation_dropout = tf.nn.dropout(mention_representation, keep_prob)
mention_representation_dropout = K.backend.dropout(mention_representation, dropout_)
left_context = context[:context_length]
right_context = context[context_length + 1:]

print 'Context placeholder created!'

input_left_context = embedding_layer(K.layers.Input(shape=(emb_dim,), dtype='float32')) # None, 300
input_right_context = embedding_layer(K.layers.Input(shape=(emb_dim,), dtype='float32')) # None, 300

print 'input_LR_context created!'

# if --attentive (LSTM + Attentions)
left_oneLSTM = K.layers.recurrent.LSTM(lstm_dim, return_sequences=True) # stateful=True, 
right_oneLSTM = K.layers.recurrent.LSTM(lstm_dim, return_sequences=True, go_backwards=True) # stateful=True, 
left_biLSTM = K.layers.wrappers.Bidirectional(left_oneLSTM, merge_mode='concat')(input_left_context)
right_biLSTM = K.layers.wrappers.Bidirectional(right_oneLSTM, merge_mode='concat')(input_right_context)

print 'biLSTM created!'

# Updating model
model.add(left_biLSTM)
model.add(Dropout(0.5))
model.add(right_biLSTM)
model.add(Dropout(0.5))

print 'biLSTM added to model!'

sys.exit()

# Attentive Encoder (example)
# activations = LSTM(units, return_sequences=True)(embedded)
# 
# compute importance for each step
# attention = Dense(1, activation='tanh')(activations)
# attention = Flatten()(attention)
# attention = Activation('softmax')(attention)
# attention = RepeatVector(units)(attention)
# attention = Permute([2, 1])(attention)
# 
# sent_representation = merge([activations, attention], mode='mul')

# context_representation, attentions = attentive_sum(left_biLSTM + right_biLSTM, input_dim=lstm_dim * 2, hidden_dim=attention_dim)

merge_biLSTM = merge([left_biLSTM, right_biLSTM], mode='sum')

attention = Dense(attention_dim, activation='tanh', input_shape=lstm_dim*2)(merge_biLSTM)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(lstm_dim)(attention)
attention = Permute([2, 1])(attention)

context_representation = merge([activations, attention], mode='mul')

# Missing --feature part...
# ...
representation = K.layers.merge.Concatenate([mention_representation_dropout, context_representation], axis=1)

# Missing --hier part...
# ...
W = weight_variable('hier_W', (rep_dim, target_dim))
logit = K.layers.merge.Dot(representation, W)

distribution = K.sigmoid(logit)

# loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logit, self.target))
loss_f = np.mean(K.backend.binary_crossentropy(logit, target, from_logits=True))
optimizer_adam = K.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer_adam, metrics=model_metrics, loss=loss_f, batch_size=1000)

# Model summary
print model.summary()

# To be continued...
# for e in range(nb_epoch):
#     print "epoch %d" % e

#     for X_train, Y_train in BatchGenerator(): 
#         model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1)

# model.fit(DATASET_X, DATASET_Y, nb_epoch=nb_epochs, batch_size=batch_size)