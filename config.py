import os
import sys
from pathlib import Path

# Set working directory to script location
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
coco_dir = script_dir / 'COCO'
iux_dir = script_dir / 'IU X-ray'

DATASET_NAME = 'COCO'

ALL_DIRECTORIES = {

    'COCO':{

        # Training data preparation
        'IMAGES_TRAIN': coco_dir / 'train2014',
        'ANNOTATIONS_TRAIN': coco_dir / 'annotations/captions_train2014.json',

        # Validation data
        'IMAGES_VAL': coco_dir / 'val2014',
        'ANNOTATIONS_VAL': coco_dir / 'annotations/captions_val2014.json',

        # Cached image features
        'IMAGE_FEATURES_TRAIN': coco_dir / 'image_features_train',
        'IMAGE_FEATURES_VAL': coco_dir / 'image_features_val',

        # Data saved during training
        # CHECKPOINT_PATH = coco_dir / 'checkpoints/train'
        #MODELS_PATH = coco_dir + '/saved_models/'
        #RESULTS_PATH = coco_dir + '/search_results/'
        'GRID_SEARCHS': coco_dir / 'grid_searchs'
        },

    'IU X-ray':{

        # Training data preparation
        'IMAGES': iux_dir / 'images',
        'ANNOTATIONS': iux_dir / 'image_caption_pairs.json',

        # Cached image features
        'IMAGE_FEATURES': iux_dir / 'image_features',

        # Data saved during training
        # CHECKPOINT_PATH = coco_dir / 'checkpoints/train'
        #MODELS_PATH = coco_dir + '/saved_models/'
        #RESULTS_PATH = coco_dir + '/search_results/'
        'GRID_SEARCHS': iux_dir / 'grid_searchs'
        }

}

DIRECTORIES = ALL_DIRECTORIES[DATASET_NAME]


if __name__ == '__main__':

    for directory in DIRECTORIES.values():
        if not directory.exists():
            directory.mkdir(parents=True)
