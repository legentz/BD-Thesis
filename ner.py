# -*- coding: utf-8 -*- 

import datetime
from model import KerasModel
from loader import Loader
from batcher import Batcher
from hook import acc_hook, save_predictions 
from sys import exit
from utils import keras_logo

# Config JSON
from config.config import config

# Cool feature
keras_logo()

# Load dicts and datasets
dicts, train_dataset, dev_dataset, test_dataset = Loader(
    paths=config['data']
).get_data()

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

model_wrapper = KerasModel(
    hyper=config['hyper']
)
model_wrapper.compile_model()

# Summary 
# TODO: apply improved names to the layers
print model_wrapper.get_model_summary()

results = model_wrapper.train_model(
    train_batcher,
    steps_per_epoch=config['train']['steps_per_epoch'],
    epochs=config['train']['epochs'],
    shuffle=config['train']['shuffle'],
    verbose=config['train']['verbose']
)

# Saving model as HDF5 model
model_wrapper.save_to_json(
    json_path=config['save_as']['name'],
    weights_path=config['save_as']['weights']
)

# Coming soon...
# model.load_from_json_and_compile()

# Prediction
# Preparing batcher...
results = model_wrapper.get_predictions(
    test_batcher,
    batch_size=config['predict']['batch_size'],
    acc_hook=config['predict']['acc_hook'],
    id2label=dicts['id2label'],
    show_results_vector=config['predict']['show_results_vector'],
    save_as_txt=config['predict']['save_as_txt'],
    verbose=config['predict']['verbose'],
)

# TODO: Solve Lambda layer error during model loading
# model_wrapper = KerasModel(load_model={
#     'json_path': 'model_saved.json',
#     'metrics': ['accuracy'],
#     'loss': 'binary_crossentropy',
#     'optimizer': 'adam',
#     'weights_path': 'model_saved_weights.h5'
# })
