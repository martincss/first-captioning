import tensorflow as tf
import time
import datetime
import sys
import logging
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping

from train_data_preparation import tokenizer, dataset_train, train_max_length
from valid_data_preparation import dataset_val
from model import Captioner
from train_utils import make_optimizer, LoggerCallback
from metrics import BLEUMetric, METEORMetric

from params import BATCH_SIZE, EPOCHS, num_examples, num_examples_val, \
                   vocab_size, VALID_BATCH_SIZE, attention_features_shape
# from config import CHECKPOINT_PATH



def padded_cross_entropy(real, pred):
    """

    Params:
        real: tensor of shape (batch_size,)
            contains the word indices for each caption word on the batch

        pred: tensor of shape (batch_size, vocab_size)
            contains logits distribution on the whole vocabulary for each word
            on the batch

    """

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train(hparams, models_path = './'):
    """

    Returns:
        results: dict
            dictionary containing model identifier, elapsed_time per epoch,
            learning curve with loss and metrics

    """

    model_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    captioner = Captioner(**hparams['model'],
                          vocab_size = vocab_size,
                          tokenizer = tokenizer,
                          batch_size = BATCH_SIZE,
                          caption_length = train_max_length,
                          valid_batch_size = VALID_BATCH_SIZE,
                          num_examples_val = num_examples_val)

    optimizer = make_optimizer(**hparams['optimizer'])

    metrics = [BLEUMetric(n_gram=1, name = 'bleu-1'),
               BLEUMetric(n_gram=2, name = 'bleu-2'),
               BLEUMetric(n_gram=3, name = 'bleu-3'),
               BLEUMetric(n_gram=4, name = 'bleu-4'),
               METEORMetric(name = 'meteor')]

    captioner.compile(optimizer, loss_fn = padded_cross_entropy,
                      metrics = metrics, run_eagerly = True)

    logger_cb = LoggerCallback()
    early_stopping_cb = EarlyStopping(monitor = 'val_bleu-4', patience = 10,
                                      mode = 'max',
                                      restore_best_weights = True)

    logging.info('Training start for model ' + model_id)
    logging.info('hparams: ' + str(hparams))

    history = captioner.fit(dataset_train, epochs=EPOCHS,
                            validation_data = dataset_val,
                            validation_steps = num_examples_val//VALID_BATCH_SIZE,
                            callbacks=[logger_cb, early_stopping_cb])

    losses = {key:value for key, value in history.history.items() if 'val' not in key}
    metrics = {key[4:]:value for key, value in history.history.items() if 'val' not in key}

    results = { 'id': model_id,
                'losses': losses,
                'epoch_times': logger_cb.epoch_times,
                'total_time': logger_cb.total_time,
                'params': captioner.count_params(),
                'instances_train': num_examples,
                'instances_valid': num_examples_val,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'vocabulary': vocab_size,
                'valid_batch_size': VALID_BATCH_SIZE,
                'valid_epoch_times': logger_cb.validation_times,
                'metrics': metrics
                }

    captioner.save_weights(str(models_path / (model_id + '.h5')))

    return results
