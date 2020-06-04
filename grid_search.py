import numpy as np
from sklearn.model_selection import ParameterGrid
import json
from nltk.translate.bleu_score import sentence_bleu

from training import train
from valid_data_preparation import img_paths_val, val_captions
from evaluation import generate_captions_all

from params import BATCH_SIZE, MODELS_PATH, RESULTS_PATH
from hyperparameters_space import grid

def split_hparams(hparams):

    hparams_by_type = {}
    hparams_by_type['encoder'] = {'embedding_dim': hparams['embedding_dim']}
    hparams_by_type['decoder'] = {'embedding_dim': hparams['embedding_dim'],
                                  'units': hparams['units']}

    return hparams_by_type


for hparams in ParameterGrid(grid):

    train_results, models = train(hparams)
    encoder, decoder = models

    predictions = generate_captions_all(img_paths_val, encoder, decoder)

    scores = []
    for prediction, reference in zip(predictions, val_captions):
        scores.append(sentence_bleu(references=[reference], hypothesis=prediction))

    results = {**hparams, **train_results}
    results['bleu-4'] = np.mean(scores)
    fname =  RESULTS_PATH + 'results_' + results['id'] + '.json'

    with open(fname, 'w') as f:
        json.dump(results, f)
