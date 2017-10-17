# -*- coding: utf-8 -*- 

import datetime, argparse
from model import KerasModel
from loader import Loader
from batcher import Batcher
from hook import acc_hook, save_predictions 
from sys import exit
from utils import print_centered

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

def init_model_wrapper():
    model_wrapper = KerasModel(hyper=config['hyper'])
    model_wrapper.compile_model()

    return model_wrapper

def get_batchers():
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

def train_model(batcher):
    return model_wrapper.train_model(
        batcher,
        steps_per_epoch=config['train']['steps_per_epoch'],
        epochs=config['train']['epochs'],
        shuffle=config['train']['shuffle'],
        verbose=config['train']['verbose']
    )

def save_model():
    model_wrapper.save_model(
        # json_path=config['save_as']['name'],
        weights_path=config['save_as']['weights']
    )

def predict_and_evaluate(batcher):
    return model_wrapper.predict_and_evaluate_model(
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

def print_model_summary(model_wrapper):
    summary = model_wrapper.get_model_summary(print_fn=print_centered)

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
        print_model_summary(model_wrapper)

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

