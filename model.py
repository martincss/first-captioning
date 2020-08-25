import tensorflow as tf
from tensorflow.nn import tanh, softmax, sigmoid
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Activation, Concatenate
from tensorflow.keras.regularizers import l1_l2

from tensorflow.keras.mixed_precision import experimental as mixed_precision

from params import feature_vector_shape, attention_features_shape, USE_FLOAT16

if USE_FLOAT16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


def get_attention(units, lstm_units, p_dropout = 0, l1_reg = 0, l2_reg = 0):

    W1 = Dense(units, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'W_feats')
    W2 = Dense(units, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'W_hidden')
    V = Dense(1, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'V')
    f_beta = Dense(1, kernel_regularizer=l1_l2(l1_reg, l2_reg), name = 'f_beta')
    dropout = Dropout(p_dropout, name = 'dropout')

    # shape = (batch, attn_features, features_shape)
    encoder_output = Input(feature_vector_shape, name = 'image_features')
    # shape = (batch, lstm_units)
    hidden_last = Input(lstm_units, name = 'last_hidden_state')

    # shape = (batch, attn_features, 1)
    score = V(tanh(
                    dropout(W1(encoder_output)) + \
                    dropout(W2(tf.expand_dims(hidden_last, axis = 1)))
                    ))
    # shape = (batch, attn_features)
    score = dropout(tf.reduce_sum(score, axis=2))
    attention_weights = Activation('softmax', dtype = 'float32')(score)

    # beta = f_beta(dropout(tf.expand_dims(hidden_last, axis = 1)))
    # shape = (batch, 1)
    beta = f_beta(hidden_last)
    beta = Activation('sigmoid', dtype='float32')(beta)

    # shape = (batch, attn_features, features_shape)
    context_vector = tf.expand_dims(attention_weights, axis=2) * encoder_output
    # shape = (batch, features_shape)
    context_vector = beta * tf.reduce_sum(context_vector, axis=1)
    context_vector = Activation('linear', dtype='float32')(context_vector)

    return Model(inputs = [encoder_output, hidden_last],
                 outputs = [context_vector, attention_weights], name = 'attention')


def get_init_h(lstm_units, n_layers, p_dropout = 0, l1_reg = 0, l2_reg = 0):

    encoder_output = Input(feature_vector_shape, name = 'image_features')
    layers = [Dense(lstm_units, activation='relu', name = 'init_h_{}'.format(i)) for i in range(n_layers)]

    h_0 = Dropout(p_dropout)(layers[0](tf.reduce_mean(encoder_output, axis=1)))

    for i in range(1, n_layers):
        h_0 = Dropout(p_dropout)(layers[i](h_0))

    h_0 = Activation('tanh', dtype='float32')(h_0)


    return Model(inputs = [encoder_output], outputs = [h_0], name = 'init_h')


def get_init_c(lstm_units, n_layers, p_dropout = 0, l1_reg = 0, l2_reg = 0):

    encoder_output = Input(feature_vector_shape, name = 'image_features')
    layers = [Dense(lstm_units, activation='relu', name = 'init_c_{}'.format(i)) for i in range(n_layers)]

    c_0 = Dropout(p_dropout)(layers[0](tf.reduce_mean(encoder_output, axis=1)))

    for i in range(1, n_layers):
        c_0 = Dropout(p_dropout)(layers[i](c_0))

    c_0 = Activation('tanh', dtype='float32')(c_0)

    return Model(inputs = [encoder_output], outputs = [c_0], name = 'init_c')



def get_decoder(embedding_dim,
                units,
                lstm_units,
                vocab_size,
                attn_dropout = 0,
                lstm_dropout =0,
                logit_dropout = 0,
                l1_reg = 0,
                l2_reg = 0):

    attention = get_attention(units, lstm_units, attn_dropout, l1_reg, l2_reg)
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                          input_length = 1, name = 'embedding')
    lstm = LSTM(lstm_units,
                   # return_sequences = True,
                   return_state = True,
                   recurrent_initializer = 'glorot_uniform',
                   dropout = lstm_dropout,
                   # recurrent_dropout = p_dropout,
                   kernel_regularizer = l1_l2(l1_reg, l2_reg))
    logits_kernel = Dense(vocab_size,
                          kernel_regularizer=l1_l2(l1_reg, l2_reg),
                          name = 'logits_kernel')
    dropout = Dropout(logit_dropout, name = 'dropout')

    # shape = (batch, 1)
    word_input = Input(1, name = 'bow_input')
    # shape = (batch, attn_features, features_shape)
    encoder_output = Input(feature_vector_shape, name = 'image_features')
    # shape = (batch, lstm_units)
    hidden_last = Input(lstm_units, name = 'last_hidden_state')
    cell_last = Input(lstm_units, name = 'last_cell_state')


    # see keras doc on Embedding layer: if input shape is (batch, input_length)
    # then output shape is (batch, input_length, embedding_dim).
    # In this case, input_length is always 1, so the embedded_word shape is
    # (batch, 1, embedding_dim). We will use this axis 1 for the lstm input
    embedded_word = embedding(word_input)

    # context_vector shape = (batch, features_shape)
    context_vector, attention_weights = attention([encoder_output, hidden_last])

    # RNN input shape must be (batch, time_steps, features). In our case,
    # time_steps is 1, so we must expand the context vector dimensions
    # lstm_output shape = (batch, lstm_units)

    lstm_input = Concatenate(axis=-1)([embedded_word,
                                       tf.expand_dims(context_vector, axis=1)])

    lstm_output, hidden, cell = lstm(lstm_input,
                                     initial_state = [hidden_last, cell_last])

    # Now we finally drop the extra 1 axis in the embedded_word

    logits_kernel_input = Concatenate(axis=-1)([tf.reduce_sum(embedded_word,
                                                            axis=1),
                                                context_vector,
                                                lstm_output])

    logits = dropout(logits_kernel(logits_kernel_input))
    # shape = (batch, vocab_size)
    logits = Activation('linear', dtype='float32')(logits)

    return Model(inputs = [word_input, encoder_output, hidden_last, cell_last],
                 outputs = [logits, attention_weights, hidden, cell])


class Captioner(Model):

    def __init__(self,
                 embedding_dim,
                 units,
                 lstm_units,
                 n_layers_init,
                 vocab_size,
                 tokenizer,
                 batch_size,
                 caption_length,
                 valid_batch_size,
                 num_examples_val,
                 init_dropout = 0,
                 attn_dropout = 0,
                 lstm_dropout =0,
                 logit_dropout = 0,
                 l1_reg = 0,
                 l2_reg = 0,
                 lambda_reg = 0.):

        super(Captioner, self).__init__()
        self.init_h = get_init_h(lstm_units, n_layers_init, init_dropout, l1_reg, l2_reg)
        self.init_c = get_init_c(lstm_units, n_layers_init, init_dropout, l1_reg, l2_reg)
        self.decoder = get_decoder(embedding_dim, units, lstm_units, vocab_size,
                                   attn_dropout, lstm_dropout, logit_dropout,
                                    l1_reg, l2_reg)
        self.lambda_reg = lambda_reg
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.caption_length = caption_length

        self.validation_steps = num_examples_val // valid_batch_size
        self.current_validation_step = 0


    def compile(self, optimizer, loss_fn, metrics, **kwargs):
        if USE_FLOAT16:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        super(Captioner, self).compile(optimizer=optimizer, loss=loss_fn, **kwargs)
        # self.optimizer = optimizer
        # self.loss_fn = loss_fn
        self.word_metrics = metrics

    @tf.function
    def train_step(self, data):

        img_tensor, target = data

        # batch_size = int(tf.shape(target)[0])
        # caption_length = int(tf.shape(target)[1])
        #
        # # batch_size, caption_length = tf.shape(target)
        batch_size = self.batch_size
        caption_length = self.caption_length
        loss = 0
        losses = {}

        with tf.GradientTape() as tape:

            word = tf.expand_dims([self.tokenizer.word_index['<start>']] * batch_size, 1)
            hidden = self.init_h(img_tensor)
            cell = self.init_c(img_tensor)

            attention_sum = tf.zeros((batch_size, attention_features_shape))

            for i in range(1, caption_length):

                predictions, attention_weights, hidden, cell = self.decoder([
                                                                    word,
                                                                    img_tensor,
                                                                    hidden,
                                                                    cell
                                                                    ],
                                                                    training=True)
                attention_sum += attention_weights

                loss += self.compiled_loss(target[:, i], predictions)
                word = tf.expand_dims(target[:, i], 1)

            losses['cross_entropy'] = loss/caption_length

            # attention regularization loss
            # loss_attn_reg = self.lambda_reg * tf.reduce_sum((1 - attention_sum)**2)
            loss_attn_reg = self.lambda_reg * tf.reduce_mean(tf.reduce_sum((1 - attention_sum)**2, axis =1))

            losses['attention_reg'] = loss_attn_reg/caption_length
            loss += loss_attn_reg

            # Weight decay losses
            loss_weight_decay = tf.add_n(self.decoder.losses)
            losses['weight_decay'] = loss_weight_decay/caption_length
            loss += loss_weight_decay

            if USE_FLOAT16:
                scaled_loss = optimizer.get_scaled_loss(loss)


        losses['total'] = loss/ caption_length

        if USE_FLOAT16:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        else:
            gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses


    def call(self, inputs, training=False, **kwargs):

        img_tensor = inputs
        # batch_size = int(tf.shape(img_tensor)[0])
        batch_size = self.valid_batch_size

        logits = []
        captions = [[] for _ in range(batch_size)]

        word = tf.expand_dims([self.tokenizer.word_index['<start>']] * batch_size, 1)
        hidden = self.init_h(img_tensor)
        cell = self.init_c(img_tensor)

        for i in range(1, self.caption_length):

            predictions, attention_weights, hidden, cell = self.decoder([
                                                                word,
                                                                img_tensor,
                                                                hidden,
                                                                cell
                                                                ], training)

            logits.append(predictions)
            word = tf.random.categorical(predictions, 1)

            for i in range(batch_size):
                next_word = self.tokenizer.index_word.get(int(word[i,0]))
                captions[i].append(next_word)

        # cut each caption up to the <end> token
        for i in range(batch_size):
            try:
                end_index = captions[i].index('<end>')

            except ValueError:
                end_index = len(captions[i])

            finally:
                captions[i] = captions[i][:end_index]

        logits = tf.stack(logits, axis=2)

        return logits, captions



    def test_step(self, data):

        # manually reset metrics (awful, I know, but wouldn't work otherwise)
        if self.current_validation_step == self.validation_steps:
            self.current_validation_step = 1
            for metric in self.word_metrics:
                metric.reset_states()

        else:
            self.current_validation_step += 1

        img_tensor, target, captions = data

        logits, captions_pred = self(img_tensor, training=False)

        # captions_true = [cap.decode('utf-8').split(' ')[1:-1] for cap in \
        #                  captions.numpy().tolist()]
        captions = self.tokenizer.sequences_to_texts(target.numpy())

        def unpad(caption):
            """
            caption:string
            """
            caption = caption.split(' ')
            end_idx = caption.index('<end>')

            return caption[1:end_idx]

        captions_true = [unpad(caption) for caption in captions]

        # self.compiled_metrics.update_state(y_true=captions_true, y_pred=captions_pred)

        for metric in self.word_metrics:
            metric.update_state(y_true=captions_true, y_pred=captions_pred)

        return {m.name: m.result() for m in self.word_metrics}
