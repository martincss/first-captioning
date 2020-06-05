import os
import sys

# Set working directory to script location
abspath = os.path.abspath(__file__)
working_directory = os.path.dirname(abspath)
os.chdir(working_directory)

IMGS_FEATURES_CACHE_DIR_TRAIN = working_directory + '/image_features_train/'
IMGS_FEATURES_CACHE_DIR_VAL = working_directory + '/image_features_val/'

# Training data preparation
IMGS_PATH_TRAIN = working_directory + '/train2014/'
ANNOTATION_FILE_TRAIN = working_directory + '/annotations/captions_train2014.json'

# Validation data
IMGS_PATH_VAL = working_directory + '/val2014/'
ANNOTATION_FILE_VAL = working_directory + '/annotations/captions_val2014.json'

# Data saved during training
CHECKPOINT_PATH = working_directory + "/checkpoints/train"
#MODELS_PATH = working_directory + '/saved_models/'
#RESULTS_PATH = working_directory + '/search_results/'
GRID_SEARCHS_PATH = working_directory + '/grid_searchs/'

if __name__ == '__main__':
    
    dirs = [IMGS_FEATURES_CACHE_DIR_TRAIN, IMGS_FEATURES_CACHE_DIR_VAL,
            CHECKPOINT_PATH, GRID_SEARCHS_PATH]

    for directory in dirs:
        if not os.path.exists(dir):
            os.mkdir(directory)
