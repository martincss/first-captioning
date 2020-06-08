from model import CNN_Encoder, RNN_Decoder
from params import BATCH_SIZE, vocab_size, features_shape, \
                   attention_features_shape


def load_encoder(fname, embedding_dim, batch_size=BATCH_SIZE,
                 features1=attention_features_shape, features2=features_shape):

    encoder = CNN_Encoder(embedding_dim)
    encoder.build((batch_size, features1, features2))
    encoder.load_weights(fname)

    return encoder


def load_decoder(fname, embedding_dim, units, batch_size=BATCH_SIZE,
                 vocab_size = vocab_size):

    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    input_shape = [(batch_size, 1),
                   (batch_size, attention_features_shape, embedding_dim),
                   (batch_size, units)]
    decoder.build(input_shape)
    decoder.load_weights(fname)

    return decoder
