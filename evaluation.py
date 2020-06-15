import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()
from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate import meteor_score

from preprocess_encode_images import extract_cache_features
from train_data_preparation import tokenizer, train_max_length

from params import attention_features_shape
from config import IMGS_FEATURES_CACHE_DIR_TRAIN, IMGS_FEATURES_CACHE_DIR_VAL


def predict_single(image, models, image_features_dir, tokenizer,
                             extract_features = False):
    """

    Parameters:
        models: tuple of Keras Model: (encoder, decoder)

    Returns:
        logits: Tensor of shape (caption_length, vocab_size), with
                caption_length variable and capped by train_max_length

        caption:

        attention_plot:


    """

    attention_plot = np.zeros((train_max_length, attention_features_shape))

    encoder, decoder = models

    hidden = decoder.reset_state(batch_size=1)

    if extract_features:
        extract_cache_features(image, image_features_dir)

    img_feature_filename = image_features_dir + \
                        image.split('/')[-1] + '.npy'

    img_tensor = np.load(img_feature_filename)


    features = encoder(img_tensor)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    logits = []
    caption = []

    for i in range(train_max_length):
        predictions, hidden, attention_weights = decoder((dec_input, features, hidden))

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        logits.append(predictions)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        caption.append(tokenizer.index_word.get(predicted_id))
        #
        # if tokenizer.index_word.get(predicted_id) == '<end>':
        #
        #     logits = tf.concat(logits, axis=0)
        #     return logits, caption, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    logits = tf.concat(logits, axis=0)

    try:
        end_index = caption.index('<end>')

    except ValueError:
        end_index = len(caption)

    finally:
        caption = caption[:end_index]

    attention_plot = attention_plot[:len(caption), :]

    return logits, caption, attention_plot


def predict_all(img_paths, models, image_features_dir, tokenizer):

    all_logits = []
    all_captions = []

    for image in img_paths:
        logits, caption, _ = predict_single(image, models, image_features_dir, tokenizer)
        all_logits.append(logits)
        all_captions.append(caption)

    return all_logits, all_captions


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




def translate_into_captions(logits, tokenizer):
    """
    Parameters:
        logits: tensor of shape (caption_length, vocab_size)

    """

    caption_length, vocab_size = logits.shape

    # using [[]]*batch_size creates multiple references to the same list
    captions = [[] for _ in range(caption_length)]

    predicted_ids = tf.random.categorical(predictions, 1).numpy()

    next_word = [tokenizer.index.get(id) for id in predicted_ids.flatten()]

    for i in range(batch_size):
        captions[i].extend(next_word[i])

    pass


def generate_captions_single(image, models, image_features_dir,
                             extract_features = False):
    """

    Parameters:
        models: tuple of Keras Model: (encoder, decoder)

    """

    attention_plot = np.zeros((train_max_length, attention_features_shape))

    encoder, decoder = models

    hidden = decoder.reset_state(batch_size=1)

    if extract_features:
        extract_cache_features(image, image_features_dir)

    img_feature_filename = image_features_dir + \
                        image.split('/')[-1] + '.npy'

    img_tensor_val = np.load(img_feature_filename)


    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(train_max_length):
        predictions, hidden, attention_weights = decoder((dec_input, features, hidden))

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word.get(predicted_id))

        if tokenizer.index_word.get(predicted_id) == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def generate_captions_all(img_paths, models, image_features_dir):

    all_captions = []

    for image in img_paths:
        caption, _ = generate_captions_single(image, models, image_features_dir)
        all_captions.append(caption)

    return all_captions


def generate_train_captions(img_paths, models):

    return generate_captions_all(img_paths, models,
                             image_features_dir = IMGS_FEATURES_CACHE_DIR_TRAIN)


def generate_valid_captions(img_paths, models):

    return generate_captions_all(img_paths, models,
                             image_features_dir = IMGS_FEATURES_CACHE_DIR_VAL)


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
