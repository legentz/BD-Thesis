#-*- coding: utf-8 -*-

# Silence Tensorflow Info and Warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import KerasModel
from loader import Loader
from batcher import Batcher

# TODO: Check for device in use (pretty-printed)
# from tensorflow.python.client import device_lib
# print device_lib.list_local_devices()

# Load dicts and datasets
dicts, train_dataset, dev_dataset, test_dataset = Loader({
    'dicts': "data/Wiki/dicts_figer.pkl",
    'train': "data/Wiki/train_figer.pkl",
    'dev': "data/Wiki/dev_figer.pkl",
    'test': "data/Wiki/test_figer.pkl"
}).get_data()

print "train_dataset_size", train_dataset["data"].shape[0]
print "dev_dataset_size", dev_dataset["data"].shape[0]
print "test_dataset_size", test_dataset["data"].shape[0]

# TODO: clearify inputs
# batch_size : 1000, context_length : 10
train_batcher = Batcher(train_dataset["storage"],train_dataset["data"],1000,10,dicts["id2vec"])
# dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],dev_dataset["data"].shape[0],10,dicts["id2vec"])
# test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],test_dataset["data"].shape[0],10,dicts["id2vec"])
test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],1,10,dicts["id2vec"])

print 'Creating the model'

# TODO: external config as JS
# TODO: change class name
model_wrapper = KerasModel(encoder='lstm', compile_model=True)
model = model_wrapper.get_model()

# TODO: This will be the last piece of the code...
# Use generate_() function to emulate the Bather class in NFGEC
# used to train huge list of samples
def generate_(train_batcher):
    while 1:
        context_data, mention_representation_data, target_data, feature_data = train_batcher.next()

        yield({
            'input_1': context_data[:,:10,:],
            'input_2': context_data[:,10+1:,:],
            'input_3': mention_representation_data
            }, {
            'output_1': target_data
            })

results = model.fit_generator(generate_(train_batcher), 2, epochs=1, shuffle=True, verbose=1) # steps_per_epoch=2000

print 'Training completed successfully'
print results

# Saving model as HDF5 model
model_wrapper.save_to_json({
    'json_path': 'model_saved.json',
    'weights_path': 'model_saved_weights.h5'
    })

print 'Model has been saved'

# Coming soon...
# model.load_from_json_and_compile()

# Prediction
context_data, mention_representation_data, target_data, feature_data = test_batcher.next()
test_to_predict = model.predict_on_batch({
            'input_1': context_data[:,:10,:],
            'input_2': context_data[:,10+1:,:],
            'input_3': mention_representation_data
            })

print test_to_predict
print target_data

# TODO: Solve Lambda layer error during model loading
# model_wrapper = KerasModel(load_model={
#     'json_path': 'model_saved.json',
#     'metrics': ['accuracy'],
#     'loss': 'binary_crossentropy',
#     'optimizer': 'adam',
#     'weights_path': 'model_saved_weights.h5'
# })
