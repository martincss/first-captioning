from functools import partial
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score


class CrossEntropyMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, logits_pred):

        metric = self.padded_cross_entropy(y_true, logits_pred)
        self.total.assign_add(metric)
        self.count.assign_add(tf.constant(1, dtype=tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    @staticmethod
    def padded_cross_entropy(real, pred):
        """

        Params:
            real: tensor of shape (batch_size,)
                contains the word indices for each caption word on the batch

            pred: tensor of shape (batch_size, vocab_size)
                contains logits distribution on the whole vocabulary for each word
                on the batch

        """
        loss_object = SparseCategoricalCrossentropy(from_logits=True,
                                                    reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)



class BLEUMetric(Metric):

    def __init__(self, n_gram, **kwargs):
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.bleu = partial(self.bleu_n, n = self.n_gram)
        self.total = self.add_weight("total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, caps_true, caps_pred):

        metric = self.bleu(caps_pred, caps_true)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(len(caps_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'n_grams': self.n_gram}

    @staticmethod
    def bleu_n(predictions, references, n):
        """

        Params:
            predictions, references: list of captions, each caption as a list of
            strings
            n: int from 1 to 4
                order of n-grams for which to compute BLEU

        """

        weights = {1: (1,), 2: (1/2, 1/2), 3:(1/3, 1/3, 1/3), 4:(1/4, 1/4, 1/4, 1/4)}

        scores = []

        for pred, ref in zip(predictions, references):

            scores.append(sentence_bleu(references=[ref], hypothesis=pred,
                                        weights = weights[n]))

        ## TODO: tidy up this shit
        if len(scores) == 1:
            return scores[0]

        return scores



class METEORMetric(Metric):

    def __init__(self, alpha=0.9, beta=3, gamma=0.5,**kwargs):
        super().__init__(**kwargs)
        self.meteor = partial(meteor_score, alpha=alpha, beta=beta, gamma=gamma)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, caps_true, caps_pred):

        metric = []
        for pred, ref in zip(caps_pred, caps_true):
            metric.append(self.meteor(references = [' '.join(ref)],
                                      hypothesis = ' '.join(pred)))

        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(len(caps_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'alpha':alpha, 'beta':beta, 'gamma':gamma}
