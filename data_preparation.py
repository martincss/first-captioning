import tensorflow as tf
import numpy as np
import json

from params import BATCH_SIZE, BUFFER_SIZE, VALID_BATCH_SIZE


def image_fnames_captions(captions_file, images_dir, partition):
    """
    Loads annotations file and return lists with each image's path and caption

    Arguments:
        partition: string
            either 'train' or 'val'

    Returns:
        all_captions: list of strings
            list with each image caption
        all_img_paths: list of paths as strings
            list with each image's path to file

    """


    with open(captions_file, 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_paths = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = images_dir + 'COCO_{}2014_'.format(partition) + \
                               '{:012d}.jpg'.format(image_id)

        all_img_paths.append(full_coco_image_path)
        all_captions.append(caption)

    return all_captions, all_img_paths


def create_dataset(image_paths, tokenized_captions, img_features_dir):
    """
    Creates a tf Dataset containing pairs extracted image features and tokenized
    caption vectors
    """

    def map_func(img_name, cap):
        img_feature_filename = img_features_dir + \
                            img_name.decode('utf-8').split('/')[-1] + '.npy'

        img_tensor = np.load(img_feature_filename)
        return img_tensor, cap


    dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_captions))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_dataset_valid(image_paths, tokenized_captions, captions,
                         img_features_dir):


    def map_func(img_name, cap_vector, caption):
        img_feature_filename = img_features_dir + \
                            img_name.decode('utf-8').split('/')[-1] + '.npy'

        img_tensor = np.load(img_feature_filename)

        # split_caption = caption.split()[1:-1]

        # return img_tensor, cap_vector, split_caption
        return img_tensor, cap_vector, caption

    # captions_tensor = tf.ragged.constant(captions)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths,
                                                  tokenized_captions,
                                                  captions))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(
              map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.string]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.batch(VALID_BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
