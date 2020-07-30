import tensorflow as tf
import time
import datetime
import sys
import logging
import numpy as np

from train_data_preparation import tokenizer, dataset_train
from valid_data_preparation import dataset_val
from model import CNN_Encoder, RNN_Decoder
from train_utils import make_optimizer
from evaluation import validation_scores

from params import BATCH_SIZE, EPOCHS, num_examples, num_examples_val, \
                   vocab_size, VALID_BATCH_SIZE, attention_features_shape
from config import CHECKPOINT_PATH


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    """

    Params:
        real: tensor of shape (batch_size,)
            contains the word indices for each caption word on the batch

        pred: tensor of shape (batch_size, vocab_size)
            contains logits distribution on the whole vocabulary for each word
            on the batch

    """
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
        models: tuple of keras Models
            the trained encoder and decoder networks


    """

    model_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    encoder = CNN_Encoder(**hparams['encoder'])
    decoder = RNN_Decoder(**hparams['decoder'], vocab_size=vocab_size)

    optimizer = make_optimizer(**hparams['optimizer'])

    lambda_reg = hparams['train']['lambda_reg']

    # ckpt = tf.train.Checkpoint(encoder=encoder,
    #                            decoder=decoder,
    #                            optimizer = optimizer)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)


    start_epoch = 0
    # if ckpt_manager.latest_checkpoint:
    #   start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    #   # restoring the latest checkpoint in checkpoint_path
    #   ckpt.restore(ckpt_manager.latest_checkpoint)

    @tf.function
    def train_step(img_tensor, target):
        loss = 0
        losses = {}

        batch_size, caption_length = target.shape

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size = batch_size)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)
        # attention_plot = tf.Variable(tf.zeros((batch_size,
        #                                      caption_length,
        #                                      attention_features_shape)))
        attention_plot = 0

        with tf.GradientTape() as tape:
            features = encoder(img_tensor, training = True)


            for i in range(1, caption_length):
                # passing the features through the decoder
                predictions, hidden, attention_weights = decoder((dec_input, features, hidden), training = True)
                attention_plot += \
                tf.reshape(attention_weights, (batch_size, attention_features_shape))

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

            losses['cross_entropy'] = loss/caption_length

            # attention regularization loss
            loss_attn_reg = lambda_reg * tf.reduce_sum((1 - attention_plot)**2)
            losses['attention_reg'] = loss_attn_reg/caption_length
            loss += loss_attn_reg

            # Weight decay losses
            loss_weight_decay = tf.add_n(encoder.losses) + tf.add_n(decoder.losses)
            losses['weight_decay'] = loss_weight_decay/caption_length
            loss += loss_weight_decay



        losses['total'] = loss/ caption_length

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, losses

    num_steps = num_examples // BATCH_SIZE

    loss_plots = {'cross_entropy':[], 'attention_reg':[], 'weight_decay':[],
                  'total':[]}
    metrics = {'cross-entropy':[], 'bleu-1':[],'bleu-2':[],'bleu-3':[],
               'bleu-4':[], 'meteor':[]}
    epoch_times = []
    val_epoch_times = []

    start = time.time()
    logging.info('Training start for model ' + model_id)
    logging.info('hparams: ' + str(hparams))
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        total_loss = {'cross_entropy':0, 'attention_reg':0, 'weight_decay':0,
                      'total':0}

        for (batch, (img_tensor, target)) in enumerate(dataset_train):
            batch_loss, t_loss = train_step(img_tensor, target)
            for key in total_loss.keys():
                total_loss[key] += float(t_loss[key])

            if batch % 100 == 0:
                logging.info('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

        # storing the epoch end loss value to plot later
        for key in loss_plots.keys():
            loss_plots[key].append(total_loss[key] / num_steps)


        # Evaluate on validation
        val_epoch_start = time.time()
        epoch_scores = validation_scores(dataset_val, (encoder, decoder), tokenizer)
        val_epoch_stop = time.time() - val_epoch_start
        val_epoch_times.append(val_epoch_stop)

        for name, score in epoch_scores.items():
            metrics[name].append(score)

        epoch_stop = time.time() - epoch_start
        epoch_times.append(epoch_stop)

        # if epoch % 1 == 0:
        #   ckpt_manager.save()

        logging.info('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss['total']/num_steps))

        logging.info('Time taken for 1 epoch {} sec\n'.format(epoch_stop))

    total_time = time.time() - start
    logging.info('Total training time: {}'.format(total_time))

    results = { 'id':model_id,
                'losses':loss_plots,
                'epoch_times':epoch_times,
                'total_time':total_time,
                'encoder_params': encoder.count_params(),
                'decoder_params': decoder.count_params(),
                'instances_train': num_examples,
                'instances_valid': num_examples_val,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'vocabulary': vocab_size,
                'valid_batch_size': VALID_BATCH_SIZE,
                'valid_epoch_times':val_epoch_times,
                'metrics_val': metrics}

    encoder.save_weights(models_path + 'encoder_' + model_id + '.h5')
    decoder.save_weights(models_path + 'decoder_' + model_id + '.h5')
    models = (encoder, decoder)

    return results, models
