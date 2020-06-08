import os
import datetime
import logging
import json
import numpy as np

from importlib import reload
from sklearn.model_selection import ParameterGrid
from nltk.translate.bleu_score import sentence_bleu

from training import train
from valid_data_preparation import img_paths_val, val_captions
from evaluation import generate_captions_all

from params import BATCH_SIZE
from config import GRID_SEARCHS_PATH
from hyperparameters_space import grid, search_name

def split_hparams(hparams):

    hparams_by_type = {}
    hparams_by_type['encoder'] = {'embedding_dim': hparams['embedding_dim']}
    hparams_by_type['decoder'] = {'embedding_dim': hparams['embedding_dim'],
                                  'units': hparams['units']}

    return hparams_by_type

def create_directories(search_name=None):

    if search_name is None:
        search_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    grid_dir = GRID_SEARCHS_PATH + search_name

    os.mkdir(grid_dir)
    os.mkdir(grid_dir + '/saved_models/')
    os.mkdir(grid_dir + '/results/')

    return grid_dir


grid_dir = create_directories(search_name)

for hparams in ParameterGrid(grid):

    # this is done because other packages use logging first
    reload(logging)
    logging.basicConfig(filename = grid_dir + '/progress.log',
                        format='%(levelname)s:%(message)s', level=logging.INFO)

    train_results, models = train(split_hparams(hparams),
                                  models_path = grid_dir + '/saved_models/')
    encoder, decoder = models

    predictions = generate_captions_all(img_paths_val, encoder, decoder)

    scores = []
    for prediction, reference in zip(predictions, val_captions):
        scores.append(sentence_bleu(references=[reference], hypothesis=prediction))

    results = {**hparams, **train_results}
    results['bleu-4'] = np.mean(scores)
    fname =  grid_dir + '/results/' + 'results_' + results['id'] + '.json'

    with open(fname, 'w') as f:
        json.dump(results, f)
