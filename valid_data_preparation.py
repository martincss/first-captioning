from utils import enable_gpu_memory_growth
from preprocess_encode_images import extract_cache_features
from preprocess_tokenize_captions import caption_features
from coco_utils import image_fnames_captions
from data_preparation import create_dataset_valid
from train_data_preparation import tokenizer, train_max_length

from params import UPDATE_CACHE, num_examples_val, top_k

from config import DIRECTORIES

enable_gpu_memory_growth()


all_captions, all_img_paths = image_fnames_captions(DIRECTORIES['ANNOTATIONS_VAL'],
                                                    DIRECTORIES['IMAGES_VAL'],
                                                    partition = 'val')

val_captions = all_captions[:num_examples_val]
img_paths_val = all_img_paths[:num_examples_val]

if UPDATE_CACHE:
    extract_cache_features(img_paths, IMGS_FEATURES_CACHE_DIR_VAL)

cap_vec_val = caption_features(val_captions, tokenizer, maxlen=train_max_length)

# val_captions = [cap.split(' ')[1:-1] for cap in val_captions]


dataset_val = create_dataset_valid(img_paths_val, cap_vec_val, val_captions, \
                            DIRECTORIES['IMAGE_FEATURES_VAL'])
