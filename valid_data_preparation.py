from utils import enable_gpu_memory_growth
from preprocess_encode_images import extract_cache_features
from preprocess_tokenize_captions import caption_features
from data_preparation import image_fnames_captions, create_dataset_valid
from train_data_preparation import tokenizer, train_max_length

from params import UPDATE_CACHE, num_examples_val, top_k

from config import ANNOTATION_FILE_VAL, IMGS_PATH_VAL, \
                   IMGS_FEATURES_CACHE_DIR_VAL

enable_gpu_memory_growth()


all_captions, all_img_paths = image_fnames_captions(ANNOTATION_FILE_VAL,
                                                    IMGS_PATH_VAL,
                                                    partition = 'val')

val_captions = all_captions[:num_examples_val]
img_paths_val = all_img_paths[:num_examples_val]

if UPDATE_CACHE:
    extract_cache_features(img_paths, IMGS_FEATURES_CACHE_DIR_VAL)

cap_vec_val = caption_features(val_captions, tokenizer, maxlen=train_max_length)

val_captions = [cap.split(' ') for cap in val_captions]


dataset_val = create_dataset_valid(img_paths_val, cap_vec_val, val_captions, \
                            IMGS_FEATURES_CACHE_DIR_VAL)
