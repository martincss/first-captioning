import os
import sys
from pathlib import Path

# Set working directory to script location
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
coco_dir = script_dir / 'COCO'


IMGS_FEATURES_CACHE_DIR_TRAIN = coco_dir / 'image_features_train'
IMGS_FEATURES_CACHE_DIR_VAL = coco_dir / 'image_features_val'

# Training data preparation
IMGS_PATH_TRAIN = coco_dir / 'train2014'
ANNOTATION_FILE_TRAIN = coco_dir / 'annotations/captions_train2014.json'

# Validation data
IMGS_PATH_VAL = coco_dir / 'val2014'
ANNOTATION_FILE_VAL = coco_dir / 'annotations/captions_val2014.json'

# Data saved during training
CHECKPOINT_PATH = coco_dir / 'checkpoints/train'
#MODELS_PATH = coco_dir + '/saved_models/'
#RESULTS_PATH = coco_dir + '/search_results/'
GRID_SEARCHS_PATH = coco_dir / 'grid_searchs'

if __name__ == '__main__':

    dirs = [IMGS_FEATURES_CACHE_DIR_TRAIN, IMGS_FEATURES_CACHE_DIR_VAL,
            CHECKPOINT_PATH, GRID_SEARCHS_PATH]

    for directory in dirs:
        if not directory.exists():
            directory.mkdir(parents=True)
