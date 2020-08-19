import time
import logging
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback

def make_optimizer(**kwargs):

    name = kwargs['optimizer']
    learning_rate = kwargs['learning_rate']

    options = {'Adam': Adam, 'RMSprop': RMSprop}

    optimizer = options[name](learning_rate)

    return optimizer


class LoggerCallback(Callback):

    def on_train_begin(self, logs = {}):

        self.train_start_time = time.time()
        self.epoch_times = []
        self.validation_times = []

    def on_epoch_begin(self, epoch, logs = {}):

        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs):

        epoch_stop = time.time() - self.epoch_start
        self.epoch_times.append(epoch_stop)
        logging.info('Epoch {} \n'.format(epoch + 1))
        logging.info(str(logs))
        logging.info('Time taken for 1 epoch {} sec\n\n'.format(epoch_stop))


    def on_test_begin(self, logs = {}):

        self.test_start = time.time()

    def on_test_end(self, logs = {}):

        self.validation_times.append(time.time() - self.test_start)

    def on_train_end(self, logs = {}):

        self.total_time = time.time() - self.train_start_time
        logging.info('Total training time: {} \n'.format(self.total_time))
