import numpy as np
from sklearn.model_selection import ParameterGrid
import json
from nltk.translate.bleu_score import sentence_bleu

from training import train
from valid_data_preparation import img_paths_val, val_captions
from evaluation import generate_captions_all

from params import BATCH_SIZE, MODELS_PATH, RESULTS_PATH

grid = [{'embedding_dim': [256, 512], 'units': [512]}]


for hparams in ParameterGrid(grid):

    results = {**hparams}

    model_id, elapsed_time, encoder, decoder = train(hparams)


    predictions = generate_captions_all(img_paths_val, encoder, decoder)

    scores = []
    for prediction, reference in zip(predictions, val_captions):
        scores.append(sentence_bleu(references=[reference], hypothesis=prediction))


    results['id'] = model_id
    results['time'] = elapsed_time
    results['bleu-4'] = np.mean(scores)
    fname =  RESULTS_PATH + 'results_' + model_id + '.json'

    with open(fname, 'w') as f:
        json.dump(results, f)
