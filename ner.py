# -*- coding: utf-8 -*- 

# Silence Tensorflow Info and Warning
import os, sys, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import KerasModel
from loader import Loader
from batcher import Batcher
from hook import acc_hook, save_predictions

print '--> Loading datasets'

# Load dicts and datasets
dicts, train_dataset, dev_dataset, test_dataset = Loader({
    'dicts': "data/Wiki/dicts_figer.pkl",
    'train': "data/Wiki/train_figer.pkl",
    'dev': "data/Wiki/dev_figer.pkl",
    'test': "data/Wiki/test_figer.pkl"
}).get_data()

print '  --> Done'

context_length = 10
batch_size = 1000

# Used to produce different backup .h5/.json
now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')

# TODO: clearify inputs when refactoring Batcher()
train_batcher = Batcher(train_dataset["storage"], train_dataset["data"], batch_size, context_length, dicts["id2vec"])
test_batcher = Batcher(test_dataset["storage"], test_dataset["data"], test_dataset['data'].shape[0], context_length, dicts["id2vec"])

print '--> Creating model'

# TODO: external config as JS
model_wrapper = KerasModel(encoder='averaging', batch_size=batch_size, context_length=context_length, compile_model=True)

print '  --> Done'
print '--> Training model'

results = model_wrapper.train_model(train_batcher, steps_per_epoch=2000, epochs=1, shuffle=True, verbose=1)

print '  --> Done'
print '--> Saving model'

# Saving model as HDF5 model
model_wrapper.save_to_json({
    'json_path': 'model_saved' + now + '.json',
    'weights_path': 'model_saved_weights' + now + '.h5'
    })

print '  --> Done'

# Coming soon...
# model.load_from_json_and_compile()

# TODO: remove get_model() before prediction
model = model_wrapper.get_model()

# Prediction
context_data, mention_representation_data, target_data, feature_data = test_batcher.next()
test_to_predict = model.predict({
                'input_1': context_data[:,:context_length,:],
                'input_2': context_data[:,context_length+1:,:],
                'input_3': mention_representation_data
            }, batch_size=1000, verbose=1)

print test_to_predict
print target_data

acc_hook(test_to_predict, target_data)
save_predictions(test_to_predict, target_data, dicts['id2label'], 'predictions' + now + '.txt')

# TODO: Solve Lambda layer error during model loading
# model_wrapper = KerasModel(load_model={
#     'json_path': 'model_saved.json',
#     'metrics': ['accuracy'],
#     'loss': 'binary_crossentropy',
#     'optimizer': 'adam',
#     'weights_path': 'model_saved_weights.h5'
# })
