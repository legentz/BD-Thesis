# Silence Tensorflow Info and Warning
import os, sys, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Cool feature!
from utils import keras_logo
keras_logo()

# AIO configuration
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
        'encoder': 'attentive',
        'lstm_dim': 100,
        'attention_dim': 100,
        'emb_dim': 300,
        'target_dim': 113,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'feature': {
            'process': True,
            'dim': 50,
            'input_dim': 70,
            'size': 600000
        },
        'hier': {
            'process': True,
            'label2id_path': './resource/Wiki/label2id_figer.txt'
        },
        'metrics': ['accuracy', 'mae'],
        'loss': 'binary_crossentropy'
    },
    'train': {
        'steps_per_epoch': 2000, # 2000
        'epochs': 5, # 5
        'shuffle': True,
        'verbose': 1
    },
    'predict': {
        'batch_size': 1000,
        'acc_hook': True,
        'show_results_vector': True,
        'verbose': 1,
        'save_as_txt': 'prediction'
    },
    'save_as': {
        'name': 'model_saved',
        'weights': 'model_saved_weights'
    }
}