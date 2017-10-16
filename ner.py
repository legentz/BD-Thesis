# -*- coding: utf-8 -*- 

from model import KerasModel
from loader import Loader
from batcher import Batcher
from config.config import config
from hook import acc_hook, save_predictions 
import datetime

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

model_wrapper = KerasModel(hyper=config['hyper'])
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
model_wrapper.save_to_json({
    'json_path': config['save_as']['name'], 
    'weights_path': config['save_as']['weights']
    })

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

# Used to produce different backup .h5/.json
# TODO: Remove from here
now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')

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
