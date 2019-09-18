import time
import torch

from utils import Namespace, caller_source

config = {
    'Data': { # data setting
        'relative_path': "data/EntityMentionRelation",
        'train_path': "conll04.corp_1_test.corp",
        'valid_path': "conll04.corp_2_test.corp"
    },
    'Model': { # model setting
        'embedding_dim': 8,
        'ngram': 5,
        'dropout': 0.35,
        'activation': torch.nn.ReLU(),
        'max_distance': 64,
        'distance_emb_size': 8,
        'rnn': {
            'layers': 2,
            'bidirectional': True,
        },
        'compact': {
            'layers': [48,],
        },
        'relemb':{
            'emb_size': 256,
        },
        'pretrained_files': {
            'word': 'data/glove.6B/glove.6B.50d.txt'
        },
        'pretrained_dims': {
            'word': 50
        },
        'graph': {
            'balance_factor': 1.5,
            'label_smoothing': 0.01,
            'focal_gamma': 2,
            'inference_interval': 5,
            'inference_training_set': False
        }
    },
    'Train': {
        'pretrained_files': {
            'word': 'data/glove.6B/glove.6B.50d.txt'
        },
        'trainer': {
            'num_epochs': 20,
            'patience': None,
            'serialization_dir': 'log.{}'.format(time.strftime("%Y%m%d-%H%M%S", time.gmtime())),
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'scheduler': {
            'type': 'reduce_on_plateau',
            'patience': 10
        },
        'iterator': {
            'batch_size': 8,
        }
    }
}

Config = Namespace(config)
