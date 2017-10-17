# -*- coding: utf-8 -*- 

import datetime
from model import KerasModel
from loader import Loader
from batcher import Batcher
from hook import acc_hook, save_predictions 
from sys import exit
import argparse

# Config JSON
from config.config import config

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load-model-weights',
        '-LW',
        dest='load_model_weights',
        help='Load model weights from a (previously saved) .h5 file'
    )
    parser.add_argument(
        '--model-summary',
        '-S',
        dest='model_summary',
        action='store_true',
        help='Print model summary after compilation'
    )
    parser.add_argument(
        '--save-model-weights',
        '-SW',
        dest='save_model_weights',
        action='store_true',
        help='Save model weights after training (into a .h5 file)'
    )
    parser.add_argument(
        '--predict-and-evaluate',
        '-P',
        dest='predict_and_evaluate',
        action='store_true',
        help='Get predictions from the test dataset and its F1-score'
    )
    parser.set_defaults(load_model_weights=False)
    parser.set_defaults(model_summary=False)
    parser.set_defaults(save_model_weights=False)
    parser.set_defaults(predict_and_evaluate=False)

    return parser.parse_args()

# Load dicts and datasets
# dicts, train_dataset, dev_dataset, test_dataset = Loader(
#     paths=config['data']
# ).get_data()

def init_model_wrapper():
    model_wrapper = KerasModel(hyper=config['hyper'])
    model_wrapper.compile_model()

    return model_wrapper

def get_batchers():
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

    return train_batcher, test_batcher

# model_wrapper = KerasModel(
#     hyper=config['hyper']
# )

# if args.load_model_weights:
#     # model_wrapper.load_from_json_and_compile(

#     # )
#     model_wrapper.compile_model()
#     # model = model_wrapper.get_model()
#     # model.load_weights(args.weights_h5)
#     # model_wrapper.set_model(model)
#     model_wrapper.set_model_weights(args.load_model_weights)
#     results = model_wrapper.get_predictions(
#         test_batcher,
#         batch_size=config['predict']['batch_size'],
#         acc_hook=config['predict']['acc_hook'],
#         id2label=dicts['id2label'],
#         show_results_vector=config['predict']['show_results_vector'],
#         save_as_txt=config['predict']['save_as_txt'],
#         verbose=config['predict']['verbose'],
#     )

#     exit()

# model_wrapper.compile_model()

def train_model(batcher):
    return model_wrapper.train_model(
        batcher,
        steps_per_epoch=config['train']['steps_per_epoch'],
        epochs=config['train']['epochs'],
        shuffle=config['train']['shuffle'],
        verbose=config['train']['verbose']
    )

# Saving model as HDF5 model
def save_model():
    model_wrapper.save_to_json(
        json_path=config['save_as']['name'],
        weights_path=config['save_as']['weights']
    )

# Summary 
# TODO: apply improved names to the layers
# print model_wrapper.get_model_summary()

# results = model_wrapper.train_model(
#     train_batcher,
#     steps_per_epoch=config['train']['steps_per_epoch'],
#     epochs=config['train']['epochs'],
#     shuffle=config['train']['shuffle'],
#     verbose=config['train']['verbose']
# )

# # Saving model as HDF5 model
# model_wrapper.save_to_json(
#     json_path=config['save_as']['name'],
#     weights_path=config['save_as']['weights']
# )

# Coming soon...
# model.load_from_json_and_compile()

# Prediction
# Preparing batcher...
def predict_and_evaluate(batcher):
    return model_wrapper.get_predictions(
        batcher,
        batch_size=config['predict']['batch_size'],
        acc_hook=config['predict']['acc_hook'],
        id2label=dicts['id2label'],
        show_results_vector=config['predict']['show_results_vector'],
        save_as_txt=config['predict']['save_as_txt'],
        verbose=config['predict']['verbose'],
    )

def get_dicts_and_datasets():
    data_loader = Loader(paths=config['data'])
    dicts, train_dataset, dev_dataset, test_dataset = data_loader.get_data()

    return dicts, train_dataset, test_dataset

# TODO: Solve Lambda layer error during model loading
# model_wrapper = KerasModel(load_model={
#     'json_path': 'model_saved.json',
#     'metrics': ['accuracy'],
#     'loss': 'binary_crossentropy',
#     'optimizer': 'adam',
#     'weights_path': 'model_saved_weights.h5'
# })

if __name__ == '__main__':

    # Process terminal inputs
    args = process_args()

    # Get dicts, datasets and train/test data batchers
    dicts, train_dataset, test_dataset = get_dicts_and_datasets()
    train_batcher, test_batcher = get_batchers()

    # Init model_wrapper
    model_wrapper = init_model_wrapper()

    # Print model summary
    if args.model_summary:
        print model_wrapper.get_model_summary()

    # Load model weights from .h5, or...
    if args.load_model_weights:
        model_wrapper.set_model_weights(args.load_model_weights)

    # ...train the model
    else:
        history = train_model(train_batcher)

        # Save model weights after training
        if args.save_model_weights:
            save_model()

    # Predict and evaluate (F1-Score)
    if args.predict_and_evaluate:
        results = predict_and_evaluate(test_batcher)

