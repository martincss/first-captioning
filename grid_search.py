import os
import datetime
import logging
import json
import numpy as np

from importlib import reload
from sklearn.model_selection import ParameterGrid

from training import train

from params import BATCH_SIZE
from config import GRID_SEARCHS_PATH
from hyperparameters_space import grid, search_name

def split_hparams(hparams):
    """
    Takes a dictionary of hyperparameters as provided by ParameterGrid and
    segments it into groups according to their usage.

    Params:
        hparams: dict with each key containing a single value

    Returns:
        hparams_by_type: dict of dicts. Each item is a dictionary containing
        the hyperparameters regarding their use (e.g. encoder, optimizer, etc.)

    """

    hparams_by_type = {}
    hparams_by_type['encoder'] = {'embedding_dim': hparams['embedding_dim'],
                                  'p_dropout':hparams['p_dropout'],
                                  'l1_reg':hparams['l1_reg'],
                                  'l2_reg':hparams['l2_reg']}
    hparams_by_type['decoder'] = {'embedding_dim': hparams['embedding_dim'],
                                  'units': hparams['units'],
                                  'p_dropout':hparams['p_dropout'],
                                  'l1_reg':hparams['l1_reg'],
                                  'l2_reg':hparams['l2_reg']}
    hparams_by_type['optimizer'] = {'optimizer': hparams['optimizer'],
                                    'learning_rate': hparams['learning_rate']}
    hparams_by_type['train'] = {'lambda_reg': hparams['lambda_reg']}

    return hparams_by_type

def create_directories(search_name=None):

    if search_name is None:
        search_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    grid_dir = GRID_SEARCHS_PATH / search_name

    if not os.path.exists(grid_dir):
        grid_dir.mkdir()
        (grid_dir / 'saved_models').mkdir()
        (grid_dir / 'results').mkdir()

    return grid_dir


grid_dir = create_directories(search_name)

for hparams in ParameterGrid(grid):

    # this is done because other packages use logging first
    reload(logging)
    logging.basicConfig(filename = grid_dir / 'progress.log',
                        format='%(levelname)s:%(message)s', level=logging.INFO)

    train_results, models = train(split_hparams(hparams),
                                  models_path = grid_dir / 'saved_models')
    encoder, decoder = models

    results = {**split_hparams(hparams), **train_results}
    fname =  grid_dir / 'results' / ('results_' + results['id'] + '.json')

    with open(fname, 'w') as f:
        json.dump(results, f)
