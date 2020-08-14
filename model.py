import tensorflow as tf
from tensorflow.nn import tanh, softmax, sigmoid
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, GRU
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
                    dropout(W1(encoder_output)) + \
                    dropout(W2(tf.expand_dims(hidden_last, axis = 1)))
                    ))

    attention_weights = softmax(dropout(score), axis=1)

    beta = dropout(f_beta(tf.expand_dims(hidden_last, axis = 1)))

    context_vector = beta * attention_weights * encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    attention_weights = tf.reduce_sum(attention_weights, axis=2)

    return Model(inputs = [encoder_output, hidden_last],
                 outputs = [context_vector, attention_weights], name = 'attention')


def get_init_h(units, p_dropout = 0, l1_reg = 0, l2_reg = 0):

    encoder_output = Input(feature_vector_shape, name = 'image_features')
    init_h = Dense(units, name = 'init_h')

    h_0 = Dropout(p_dropout)(init_h(tf.reduce_mean(encoder_output, axis=1)))

    return Model(inputs = [encoder_output], outputs = [h_0], name = 'init_h')


def get_init_c(units, p_dropout = 0, l1_reg = 0, l2_reg = 0):

    encoder_output = Input(feature_vector_shape, name = 'image_features')
    init_h = Dense(units, name = 'init_c')

    c_0 = Dropout(p_dropout)(init_h(tf.reduce_mean(encoder_output, axis=1)))

    return Model(inputs = [encoder_output], outputs = [c_0], name = 'init_c')



def get_decoder(embedding_dim,
                units,
                vocab_size,
                p_dropout = 0,
                l1_reg = 0,
                l2_reg = 0):

    attention = get_attention(units, p_dropout, l1_reg, l2_reg)
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                          input_length = 1, name = 'embedding')
    lstm = LSTM(units,
                   # return_sequences = True,
                   return_state = True,
                   recurrent_initializer = 'glorot_uniform',
                   dropout = p_dropout,
                   # recurrent_dropout = p_dropout,
                   kernel_regularizer = l1_l2(l1_reg, l2_reg))
    logits_kernel = Dense(vocab_size,
                          kernel_regularizer=l1_l2(l1_reg, l2_reg),
                          name = 'logits_kernel')
    dropout = Dropout(p_dropout, name = 'dropout')


    word_input = Input(1, name = 'bow_input')
    encoder_output = Input(feature_vector_shape, name = 'image_features')
    hidden_last = Input(units, name = 'last_hidden_state')
    cell_last = Input(units, name = 'last_cell_state')


    # see keras doc on Embedding layer: if input shape is (batch, input_length)
    # then output shape is (batch, input_length, embedding_dim).
    # In this case, input_length is always 1, so the embedded_word shape is
    # (batch, 1, embedding_dim). We will use this axis 1 for the lstm input
    embedded_word = embedding(word_input)

    context_vector, attention_weights = attention([encoder_output, hidden_last])

    # RNN input shape must be (batch, time_steps, features). In our case,
    # time_steps is 1, so we must expand the context vector dimensions
    lstm_output, hidden, cell = lstm(tf.concat([
                                        embedded_word,
                                        tf.expand_dims(context_vector, axis=1)],
                                        axis = -1),
                                initial_state = [hidden_last, cell_last])

    # Now we finally drop the extra 1 axis in the embedded_word
    logits = dropout(logits_kernel(tf.concat([tf.reduce_sum(embedded_word,
                                                            axis=1),
                                              context_vector,
                                              lstm_output],
                                              axis=-1)))

    return Model(inputs = [word_input, encoder_output, hidden_last, cell_last],
                 outputs = [logits, attention_weights, hidden, cell])


class Captioner(Model):

    def __init__(self,
                 embedding_dim,
                 units,
                 vocab_size,
                 tokenizer,
                 p_dropout = 0,
                 l1_reg = 0,
                 l2_reg = 0,
                 lambda_reg = 0.):

        super(Captioner, self).__init__()
        self.init_h = get_init_h(units, p_dropout , l1_reg, l2_reg)
        self.init_c = get_init_c(units, p_dropout, l1_reg, l2_reg)
        self.decoder = get_decoder(embedding_dim, units, vocab_size, p_dropout,
                                    l1_reg, l2_reg)
        self.lambda_reg = lambda_reg
        self.tokenizer = tokenizer


    def compile(self, optimizer, loss_fn, metrics):
        super(Decoder, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics


    def train_step(self, data):

        img_tensor, target = data
        batch_size, caption_length = target.shape

        loss = 0
        losses = {}

        with tf.GradientTape() as tape:

            word = tf.expand_dims([self.tokenizer.word_index['<start>']] * batch_size, 1)
            hidden = self.init_h(img_tensor)
            cell = self.init_c(img_tensor)

            attention_sum = tf.zeros((batch_size, attention_features_shape, 1))

            for i in range(1, caption_length):

                predictions, attention_weights, hidden, cell = self.decoder([
                                                                    word,
                                                                    img_tensor,
                                                                    hidden,
                                                                    cell
                                                                    ])
                # attention_sum += attention_weights

                loss += loss_function(target[:, i], predictions)

            losses['cross_entropy'] = loss/caption_length

            # attention regularization loss
            loss_attn_reg = self.lambda_reg * tf.reduce_sum((1 - attention_sum)**2)
            losses['attention_reg'] = loss_attn_reg/caption_length
            loss += loss_attn_reg

            # Weight decay losses
            loss_weight_decay = tf.add_n(self.decoder.losses)
            losses['weight_decay'] = loss_weight_decay/caption_length
            loss += loss_weight_decay


        losses['total'] = loss/ caption_length

        gradients = tape.gradient(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        return losses



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
