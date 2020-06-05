import tensorflow as tf
import numpy as np
from PIL import Image

from train_data_preparation import tokenizer, train_max_length

from params import attention_features_shape
from config import IMGS_FEATURES_CACHE_DIR_VAL


def generate_captions_single(image, encoder, decoder):
    attention_plot = np.zeros((train_max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    img_feature_filename = IMGS_FEATURES_CACHE_DIR_VAL + \
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


def generate_captions_all(img_paths, encoder, decoder):

    all_captions = []

    for image in img_paths:
        caption, _ = generate_captions_single(image, encoder, decoder)
        all_captions.append(caption)

    return all_captions


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
