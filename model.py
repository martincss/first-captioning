import tensorflow as tf
from tensorflow.nn import tanh, softmax, sigmoid
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, GRU
from tensorflow.keras.regularizers import l1_l2

from params import feature_vector_shape


def get_attention(units, p_dropout = 0, l1_reg = 0, l2_reg = 0):

    W1 = Dense(units, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'W_feats')
    W2 = Dense(units, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'W_hidden')
    V = Dense(1, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'V')
    f_beta = Dense(1, kernel_regularizer=l1_l2(l1_reg, l2_reg), activation='sigmoid', name = 'f_beta')
    dropout = Dropout(p_dropout, name = 'dropout')

    encoder_output = Input(feature_vector_shape, name = 'image_features')
    hidden_last = Input(units, name = 'last_hidden_state')

    score = V(tanh(
                (dropout(W1(encoder_output)) + \
                 dropout(W2(tf.expand_dims(hidden_last, axis = 1)))
                 )))

    attention_weights = softmax(dropout(score), axis=1)

    beta = dropout(f_beta(hidden_last))

    context_vector = beta * attention_weights * encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return Model(inputs = [encoder_output, hidden_last],
                 outputs = [context_vector, attention_weights])

def get_decoder():
    pass

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, p_dropout = 0, l1_reg = 0, l2_reg = 0):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        self.W2 = Dense(units, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        self.V = Dense(1, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        self.dropout = Dropout(p_dropout)

    # def build(self, batch_input_shape):


    def call(self, features, hidden, training):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(
                            self.dropout(self.W1(features), training) + \
                            self.dropout(self.W2(hidden_with_time_axis), training))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.dropout(self.V(score), training), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim, p_dropout = 0, l1_reg = 0, l2_reg = 0):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = Dense(embedding_dim, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        self.dropout = Dropout(p_dropout)

    # def build(self, batch_input_shape):
    #     super().build(batch_input_shape)

    def call(self, x, training):
        x = self.fc(x)
        x = self.dropout(x, training)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self,
                 embedding_dim,
                 units,
                 vocab_size,
                 p_dropout = 0,
                 l1_reg = 0,
                 l2_reg = 0):

        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.units,
                       return_sequences = True,
                       return_state = True,
                       recurrent_initializer = 'glorot_uniform',
                       dropout = p_dropout,
                       # recurrent_dropout = p_dropout,
                       kernel_regularizer = l1_l2(l1_reg, l2_reg))
        self.fc1 = Dense(self.units, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        self.fc2 = Dense(vocab_size, kernel_regularizer=l1_l2(l1_reg, l2_reg))

        self.attention = BahdanauAttention(units=units,
                                           p_dropout = p_dropout,
                                           l1_reg = l1_reg,
                                           l2_reg = l2_reg)
        self.dropout = Dropout(p_dropout)

    def call(self, inputs, training):

        x, features, hidden = inputs

        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden, training)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, training = training)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        x = self.dropout(x, training)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        x = self.dropout(x, training)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
