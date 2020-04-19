import tensorflow as tf
import numpy as np
from tqdm import tqdm

from params import CACHE_FEATURES_BATCH_SIZE, UPDATE_CACHE

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def image_features_extracter():

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model


def extract_cache_features(img_name_vector):

    if UPDATE_CACHE:

        image_features_extract_model = image_features_extracter()


        # Get unique images
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
          load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
                                                    CACHE_FEATURES_BATCH_SIZE)

        for img, path in tqdm(image_dataset):
          batch_features = image_features_extract_model(img)
          batch_features = tf.reshape(batch_features,
                                      (batch_features.shape[0], -1, batch_features.shape[3]))

          for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
