# Silence Tensorflow Info and Warning
import os, sys, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        'feature': False,
        'hier': {
            'process': False,
            'label2id_path': './resource/Wiki/label2id_figer.txt'
        },
        'encoder': 'attentive'
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