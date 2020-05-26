import tensorflow as tf
import numpy as np
from tqdm import tqdm

from params import CACHE_FEATURES_BATCH_SIZE, UPDATE_CACHE

def load_image(image_path):
    """
    Loads and preprocesses the input image by resizing to 299x299 and
    normalizing values to the [-1,1] range.

    Args:
        image_path: string
            path of image file

    Returns:
        img: Tensor of shape (299,299,3)
            preprocessed image tensor

        image_path: string
            same as input

    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def image_features_extracter():
    """
    Creates an instance of the InceptionV3 model used for image feature
    extraction. This model is used without the top classification layers, i.e
    up to the last convolutional layer.

    See the README for details on the architecture.

    """

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    # TODO: not really sure why this redefinition is necessary (instead of
    # just using image_model directly)
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model


def extract_cache_features(img_name_vector, cache_dir):
    """
    Extracts features for each image in input vector and caches them to disk as
    numpy binaries .npy, to the specified directory.

    If the UPDATE_CACHE global is set to false, does nothing (thus enabling this
    lengthy process only when explicitly set in the config script).

    The loading and preprocessing is done by batches, with size given by the
    CACHE_FEATURES_BATCH_SIZE global.
    # TODO: understand the dataset objects here

    Assumes path of each image to be 'dir/img_name.jpg' to replace the directory
    with the passed cache directory

    Args:
        img_name_vector: list of strings
            list containing paths of image files

        cache_dir: string
            path to the directory in which to save the numpy binaries


    """

    if UPDATE_CACHE:

        image_features_extract_model = image_features_extracter()


        # Get unique images
        encode_train = sorted(set(img_name_vector))


        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
          load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
                                                    CACHE_FEATURES_BATCH_SIZE)

        for img, path in tqdm(image_dataset):
          batch_features = image_features_extract_model(img)
          batch_features = tf.reshape(batch_features,
                                      (batch_features.shape[0], -1, batch_features.shape[3]))

          for bf, p in zip(batch_features, path):
            # strips image directory from path and redirects to cache directory
            path_of_feature = p.numpy().decode("utf-8")
            path_of_feature = cache_dir + path_of_feature.split('/')[1]
            np.save(path_of_feature, bf.numpy())
