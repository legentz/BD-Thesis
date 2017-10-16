# -*- coding: utf-8 -*- 

# Silence Tensorflow Info and Warning
import os, sys, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import KerasModel
from loader import Loader
from batcher import Batcher
from hook import acc_hook, save_predictions

config = {
    'data': {
        'dicts': "data/Wiki/dicts_figer.pkl",
        'train': "data/Wiki/train_figer.pkl",
        'dev': "data/Wiki/dev_figer.pkl",
        'test': "data/Wiki/test_figer.pkl",
    },
    'hyper': {
        'batch_size': 1000,
        'context_length': 10,
        'feature': True,
        'hier': True,
        'encoder': 'averaging'
    },
    'train': {
        'steps_per_epoch': 2000,
        'epochs': 5,
        'shuffle': True,
        'verbose': 1
    },
    'predict': {
        'batch_size': 1000,
        'verbose': 1,
        'save_as_txt': 'prediction'
    },
    'save_as': {
        'name': 'model_saved',
        'weights': 'model_saved_weights'
    }
}

print '--> Loading datasets'

# Load dicts and datasets
dicts, train_dataset, dev_dataset, test_dataset = Loader(
    config['data']
).get_data()

print '  --> Done'

# Used to produce different backup .h5/.json
now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')

# TODO: clearify inputs when refactoring Batcher()
train_batcher = Batcher(
    train_dataset["storage"],
    train_dataset["data"],
    config['hyper']['batch_size'],
    config['hyper']['context_length'],
    dicts["id2vec"]
)

test_batcher = Batcher(
    test_dataset["storage"],
    test_dataset["data"],
    test_dataset['data'].shape[0],
    config['hyper']['context_length'],
    dicts["id2vec"]
)

print '--> Creating model'

# TODO: external config as JS
model_wrapper = KerasModel(
    encoder=config['hyper']['encoder'],
    feature=config['hyper']['feature'],
    batch_size=config['hyper']['batch_size'],
    context_length=config['hyper']['context_length']
)

print '  --> Done'
print '--> Training model'

results = model_wrapper.train_model(
    train_batcher,
    steps_per_epoch=config['train']['steps_per_epoch'],
    epochs=config['train']['epochs'],
    shuffle=config['train']['shuffle'],
    verbose=config['train']['verbose']
)

print '  --> Done'
print '--> Saving model'

# Saving model as HDF5 model
model_wrapper.save_to_json({
    'json_path': config['save_as']['name'] + now + '.json', # TODO: formats, now have to be hidden 
    'weights_path': config['save_as']['weights'] + now + '.h5' # TODO: formats, now have to be hidden 
    })

print '  --> Done'

# Coming soon...
# model.load_from_json_and_compile()

# TODO: remove get_model() before prediction
model = model_wrapper.get_model()

# Prediction
# Preparing batcher...
context_data, mention_representation_data, target_data, feature_data = test_batcher.next()

# TODO: move inside KerasModel
test_to_predict = model.predict({
                'input_1': context_data[:,:config['hyper']['context_length'],:],
                'input_2': context_data[:,config['hyper']['context_length']+1:,:],
                'input_3': mention_representation_data
            }, batch_size=config['predict']['batch_size'], verbose=config['predict']['verbose'])

# Make it right...
acc_hook(test_to_predict, target_data)
save_predictions(test_to_predict, target_data, dicts['id2label'], config['predict']['save_as_txt'] + now + '.txt')

# TODO: Solve Lambda layer error during model loading
# model_wrapper = KerasModel(load_model={
#     'json_path': 'model_saved.json',
#     'metrics': ['accuracy'],
#     'loss': 'binary_crossentropy',
#     'optimizer': 'adam',
#     'weights_path': 'model_saved_weights.h5'
# })
