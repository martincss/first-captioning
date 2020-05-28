import tensorflow as tf
from model import CNN_Encoder, RNN_Decoder
from params import BATCH_SIZE, embedding_dim, units, vocab_size


def load_encoder(fname, batch_size=BATCH_SIZE, features1=64, features2=2048):

    encoder = CNN_Encoder(embedding_dim)
    encoder.build(input_shape=tf.TensorShape((batch_size, features1, features2)))
    encoder.load_weights(fname)

    return encoder


def load_decoder(fname, batch_size=BATCH_SIZE, embedding_dim = embedding_dim,
                 units = units, vocab_size = vocab_size):

    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    input_shape = [(batch_size, 1), (batch_size, 64, embedding_dim), (batch_size, units)]
    decoder.build(input_shape)
    decoder.load_weights(fname)

    return decoder
