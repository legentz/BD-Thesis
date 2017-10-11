#-*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.externals import joblib
import sys
from model import Model
from loader import Loader
from batcher import Batcher

##
## Load dicts and datasets
##
dicts, train_dataset, dev_dataset, test_dataset = Loader({
    'dicts': "data/Wiki/dicts_figer.pkl",
    'train': "data/Wiki/train_figer.pkl",
    'dev': "data/Wiki/dev_figer.pkl",
    'test': "data/Wiki/test_figer.pkl"
}).get_data()

print "train_dataset_size", train_dataset["data"].shape[0]
print "dev_dataset_size", dev_dataset["data"].shape[0]
print "test_dataset_size", test_dataset["data"].shape[0]

# batch_size : 1000, context_length : 10
train_batcher = Batcher(train_dataset["storage"],train_dataset["data"],1000,10,dicts["id2vec"])
dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],dev_dataset["data"].shape[0],10,dicts["id2vec"])
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],test_dataset["data"].shape[0],10,dicts["id2vec"])

model = Model(dev_test=True)

# model.train()

print 'End.'
# Todo: This will be the last piece of the code...
#
# Use generate_() function to emulate the Bather class in NFGEC
# used to train huge list of samples
def generate_(path):
    while 1:
        context_data, mention_representation_data, target_data, feature_data = train_batcher.next()
        # missing something ...
        representation = tf.concat(1, [mention_representation_data, context_representation])
        yield (representation, target_data) # This replaces the return statement

model.fit_generator(generate_('/my_file.txt'), samples_per_epoch=2000, nb_epoch=5, shuffle=True)
