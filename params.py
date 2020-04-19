import os

# Image preprocessing
CACHE_FEATURES_BATCH_SIZE = 4
UPDATE_CACHE = False

# Training data preparation

IMGS_PATH = os.path.abspath('.') + '/train2014/'
ANNOTATION_FILE = './annotations/captions_train2014.json'

# Select the first 30000 captions from the shuffled set
num_examples = 30000

# Choose the top 5000 words from the vocabulary
top_k = 5000


# Model
embedding_dim = 256
units = 512
vocab_size = top_k + 1


# Training

BATCH_SIZE = 16
BUFFER_SIZE = 1000
EPOCHS = 6
CHECKPOINT_PATH = "./checkpoints/train"

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
