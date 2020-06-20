from keras.metrics import Metric
from evaluation import bleu_n

class BLEUMetric(Metric):

    def __init__(self, n_gram, **kwargs):
        super().__init__(**kwargs)
        self.n_gram = n
        self.bleu = partial(bleu_n, n = self.n_gram)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, caps_true, caps_pred):

        metric = self.bleu(caps_true, caps_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(caps_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'n_grams': self.n_gram}
