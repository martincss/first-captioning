from utils import running_on_cluster

# Image preprocessing
if running_on_cluster():
    CACHE_FEATURES_BATCH_SIZE = 32
    UPDATE_CACHE = False
else:
    CACHE_FEATURES_BATCH_SIZE = 2
    UPDATE_CACHE = False

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

feature_vector_shape = (attention_features_shape, features_shape)


# Training
if running_on_cluster():
    BATCH_SIZE = 32
    EPOCHS = 20
    BUFFER_SIZE = 1000
    VALID_BATCH_SIZE = 128
else:
    BATCH_SIZE = 4
    EPOCHS = 10
    BUFFER_SIZE = 100
    VALID_BATCH_SIZE = 16


# Select the number of instances for the training set
# This will only be used as a slicing index, so a value of -1 will use the full
# set
if running_on_cluster():
    num_examples = 60000
    num_examples_val = 5000
else:
    num_examples = 3000
    num_examples_val = 100

# Ensure that all batches are the same size (the implementation doesn't work
# with a dynamic batch_size). Passing steps_per_epoch to fit won't solve it since
# next epoch picks up from last batch (whose size is smaller than batch_size)
# For some reason, validation_steps works well in fit.
num_examples = num_examples - num_examples%BATCH_SIZE

# Data preparation

# Max length of sequences, has a huge gpu memory footprint due to the tracking
# in gradient tape (when using the attention regularization)

# Choose the vocabulary size, by selecting a the top k words by order of use
# frequency
if running_on_cluster():
    top_k = 5000
    maxlen = None

else:
    top_k = 1000
    maxlen = None

vocab_size = top_k + 1
