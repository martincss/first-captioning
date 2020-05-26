import os
import sys
from utils import running_on_cluster

# Set working directory to script location
abspath = os.path.abspath(__file__)
working_directory = os.path.dirname(abspath)
os.chdir(working_directory)

# Image preprocessing
if running_on_cluster():
    CACHE_FEATURES_BATCH_SIZE = 32
    UPDATE_CACHE = True
else:
    CACHE_FEATURES_BATCH_SIZE = 4
    UPDATE_CACHE = False

IMGS_FEATURES_CACHE_DIR_TRAIN = working_directory + '/image_features_train/'
IMGS_FEATURES_CACHE_DIR_VAL = working_directory + '/image_features_val/'

# Training data preparation
IMGS_PATH_TRAIN = working_directory + '/train2014/'
ANNOTATION_FILE = working_directory + '/annotations/captions_train2014.json'

# Select the number of instances for the training set
# This will only be used as a slicing index, so a value of -1 will use the full
# set
if running_on_cluster():
    num_examples = 30000
else:
    num_examples = 1000


# Choose the top 5000 words from the vocabulary
top_k = 5000


# Model
embedding_dim = 256
units = 512
vocab_size = top_k + 1


# Training
if running_on_cluster():
    BATCH_SIZE = 64
else:
    BATCH_SIZE = 16

BUFFER_SIZE = 1000
EPOCHS = 20
CHECKPOINT_PATH = "./checkpoints/train"

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
