import tensorflow as tf
import time
import datetime
import sys
import numpy as np
from train_data_preparation import tokenizer, dataset_train
from model import CNN_Encoder, RNN_Decoder

from params import EPOCHS, CHECKPOINT_PATH, MODELS_PATH, num_examples, BATCH_SIZE


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


def train(hparams):

    embedding_dim = hparams['embedding_dim']
    units = hparams['units']
    vocab_size = hparams['vocab_size']

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

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

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []


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

    start = time.time()
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset_train):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])),
                  file=sys.stdout)
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 1 == 0:
          ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss/num_steps),
                                             file=sys.stdout)
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - epoch_start),
                file=sys.stdout)

    total_time = time.time() - start

    model_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    encoder.save_weights(MODELS_PATH + 'encoder_' + model_id + '.h5')
    decoder.save_weights(MODELS_PATH + 'decoder_' + model_id + '.h5')

    return model_id, total_time, encoder, decoder
