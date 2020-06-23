import tensorflow as tf
import numpy as np

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


def predict_batch(img_tensor_batch, models, tokenizer, train_max_length):
    """
    Predicts logits for the probability distribution of words, for the whole
    caption.

    Params:
        img_tensor_batch: tensor of shape (batch_size, 64, 2048) (latter two
            are attention_features_shape and features_shape from InceptionV3)

        models: tuple of (encoder, decoder)

        tokenizer: trained on train_captions

        train_max_length: int
            longest caption length in training dataset (all captions are padded
                to this length)

    Returns:
        logits: tensor of shape (batch_size, vocab_size, train_max_length)
            contains logits for each word and each instance on the batch
    

    """

    batch_size = img_tensor_batch.shape[0]
    encoder, decoder = models

    features_batch = encoder(img_tensor_batch)

    hidden = decoder.reset_state(batch_size = batch_size)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']]*batch_size, 1)

    logits = []
    captions = []

    # forces to predict up to caption_length
    for i in range(train_max_length):

        # not using attention_weights here
        predictions, hidden, _ = decoder((dec_input, features_batch, hidden))

        logits.append(predictions)
        predicted_id = tf.random.categorical(predictions, 1)
        # caption.append(tokenizer.index_word.get(predicted_id))

        dec_input = predicted_id

    logits = tf.stack(logits, axis=2)

    return logits




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
