import tensorflow as tf
import time
import datetime
import sys
import logging
import numpy as np

from train_data_preparation import tokenizer, dataset_train
from model import CNN_Encoder, RNN_Decoder

from params import BATCH_SIZE, EPOCHS, num_examples, num_examples_val, \
                   vocab_size
from config import CHECKPOINT_PATH


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
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

    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)


    start_epoch = 0
    # if ckpt_manager.latest_checkpoint:
    #   start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    #   # restoring the latest checkpoint in checkpoint_path
    #   ckpt.restore(ckpt_manager.latest_checkpoint)

    @tf.function
    def train_step(img_tensor, target):
      loss = 0

      # initializing the hidden state for each batch
      # because the captions are not related from image to image
      hidden = decoder.reset_state(batch_size=target.shape[0])

      dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

      with tf.GradientTape() as tape:
          features = encoder(img_tensor)

          for i in range(1, target.shape[1]):
              # passing the features through the decoder
              predictions, hidden, _ = decoder((dec_input, features, hidden))

              loss += loss_function(target[:, i], predictions)

              # using teacher forcing
              dec_input = tf.expand_dims(target[:, i], 1)

      total_loss = (loss / int(target.shape[1]))

      trainable_variables = encoder.trainable_variables + decoder.trainable_variables

      gradients = tape.gradient(loss, trainable_variables)

      optimizer.apply_gradients(zip(gradients, trainable_variables))

      return loss, total_loss

    num_steps = num_examples // BATCH_SIZE

    loss_plot = []
    epoch_times = []

    start = time.time()
    logging.info('Training start for model ' + model_id)
    logging.info('hparams: ' + str(hparams))
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset_train):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                logging.info('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

        # storing the epoch end loss value to plot later
        loss_plot.append(float(total_loss.numpy()) / num_steps)
        epoch_stop = time.time() - epoch_start
        epoch_times.append(epoch_stop)

        if epoch % 1 == 0:
          ckpt_manager.save()

        logging.info('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss/num_steps))

        logging.info('Time taken for 1 epoch {} sec\n'.format(epoch_stop))

    total_time = time.time() - start
    logging.info('Total training time: {}'.format(total_time))

    results = {'id':model_id, 'loss':loss_plot, 'time':epoch_times,
                'total_time':total_time,
                'encoder_params': encoder.count_params(),
                'decoder_params': decoder.count_params(),
                'instances_train': num_examples,
                'instances_valid': num_examples_val,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'vocabulary': vocab_size}

    encoder.save_weights(models_path + 'encoder_' + model_id + '.h5')
    decoder.save_weights(models_path + 'decoder_' + model_id + '.h5')
    models = (encoder, decoder)

    return results, models
