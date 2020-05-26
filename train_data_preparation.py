import tensorflow as tf
import numpy as np
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import json

from preprocess_encode_images import extract_cache_features
from preprocess_tokenize_captions import make_tokenizer, caption_features

from params import IMGS_PATH_TRAIN, ANNOTATION_FILE, num_examples, top_k, \
                    BUFFER_SIZE, BATCH_SIZE, IMGS_FEATURES_CACHE_DIR_TRAIN
from utils import enable_gpu_memory_growth

enable_gpu_memory_growth()

# Read the json file
with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = IMGS_PATH_TRAIN + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

extract_cache_features(img_name_vector, IMGS_FEATURES_CACHE_DIR_TRAIN)
cap_vector = caption_features(train_captions, top_k)
tokenizer = make_tokenizer(train_captions, top_k)

# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

num_steps = len(img_name_train) // BATCH_SIZE

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
