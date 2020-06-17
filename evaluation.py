import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()
from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score

from preprocess_encode_images import extract_cache_features
from train_data_preparation import tokenizer, train_max_length

from params import attention_features_shape
from config import IMGS_FEATURES_CACHE_DIR_TRAIN, IMGS_FEATURES_CACHE_DIR_VAL



def bleu_n(prediction, reference, n):

    weights = {1: (1,), 2: (1/2, 1/2), 3:(1/3, 1/3, 1/3), 4:(1/4, 1/4, 1/4, 1/4)}

    score = sentence_bleu(references=[reference], hypothesis=prediction,
                          weights = weights[n])

    return score


def all_scores_single(predicted_logits, predicted_caption, val_cap_vector,
               val_caption, lossf):

    scores= {}

    scores['cross-entropy'] = float(lossf(val_cap_vector, predicted_logits).numpy())

    for n in range(1,5):
        scores['bleu-' + str(n)] = bleu_n(reference=val_caption,
                                          prediction=predicted_caption,
                                          n=n)
    return scores


def all_scores_all(predicted_logits_all, predicted_captions, val_cap_vectors,
               val_captions, lossf):

    n_examples = len(val_captions)

    scores_accum = {'cross-entropy':[], 'bleu-1':[],'bleu-2':[],'bleu-3':[],
               'bleu-4':[]}

    for i in range(n_examples):

        scores = all_scores_single(predicted_logits_all[i],
                                   predicted_captions[i],
                                   val_cap_vectors[i],
                                   val_captions[i],
                                   lossf)

        for name, score in scores.items():
            scores_accum[name].append(score)

    return {name:np.mean(score) for name, score in scores_accum.items()}


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
