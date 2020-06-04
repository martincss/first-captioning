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
    UPDATE_CACHE = False
else:
    CACHE_FEATURES_BATCH_SIZE = 2
    UPDATE_CACHE = False

IMGS_FEATURES_CACHE_DIR_TRAIN = working_directory + '/image_features_train/'
IMGS_FEATURES_CACHE_DIR_VAL = working_directory + '/image_features_val/'

# Training data preparation
IMGS_PATH_TRAIN = working_directory + '/train2014/'
ANNOTATION_FILE_TRAIN = working_directory + '/annotations/captions_train2014.json'

# Validation data
IMGS_PATH_VAL = working_directory + '/val2014/'
ANNOTATION_FILE_VAL = working_directory + '/annotations/captions_val2014.json'

# Select the number of instances for the training set
# This will only be used as a slicing index, so a value of -1 will use the full
# set
if running_on_cluster():
    num_examples = 30000
    num_examples_val = 200
else:
    num_examples = 1000
    num_examples_val = 100


# Choose the top 5000 words from the vocabulary
if running_on_cluster:
    top_k = 5000
else:
    top_k = 50


# Model
embedding_dim = 256
units = 512
vocab_size = top_k + 1


# Training
if running_on_cluster():
    BATCH_SIZE = 64
    EPOCHS = 20
else:
    BATCH_SIZE = 16
    EPOCHS = 2

BUFFER_SIZE = 1000

CHECKPOINT_PATH = working_directory + "/checkpoints/train"
MODELS_PATH = working_directory + '/saved_models/'
RESULTS_PATH = working_directory + '/search_results/'
GRID_SEARCHS_PATH = working_directory + '/grid_searchs/'

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
