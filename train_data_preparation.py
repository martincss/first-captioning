from sklearn.utils import shuffle

from utils import enable_gpu_memory_growth
from preprocess_encode_images import extract_cache_features
from preprocess_tokenize_captions import make_tokenizer, caption_features, \
                                         calc_max_length
from coco_utils import image_fnames_captions
from data_preparation import create_dataset

from params import UPDATE_CACHE, num_examples, top_k, maxlen
from config import DIRECTORIES




enable_gpu_memory_growth()

all_captions, all_img_paths = image_fnames_captions(DIRECTORIES['ANNOTATIONS_TRAIN'],
                                                    DIRECTORIES['IMAGES_TRAIN'],
                                                    partition = 'train')

# Shuffle captions and image_names together
# Set a random state
train_captions, img_paths = shuffle(all_captions,
                                    all_img_paths,
                                    random_state=1)

train_captions = train_captions[:num_examples]
img_paths = img_paths[:num_examples]

if UPDATE_CACHE:
    extract_cache_features(img_paths, IMGS_FEATURES_CACHE_DIR_TRAIN)

tokenizer = make_tokenizer(all_captions, top_k)
cap_vector = caption_features(train_captions, tokenizer, maxlen)

# Calculates the max_length, which is used to store the attention weights
if maxlen is None:
    train_max_length = calc_max_length(tokenizer.texts_to_sequences(train_captions))
else:
    train_max_length = maxlen

# Training set already split from training and validation directories
img_paths_train, cap_train = img_paths, cap_vector


dataset_train = create_dataset(img_paths_train, cap_train, \
                               DIRECTORIES['IMAGE_FEATURES_TRAIN'])
