from tensorflow.keras.optimizers import Adam, RMSprop


def make_optimizer(**kwargs):

    name = kwargs['optimizer']
    learning_rate = kwargs['learning_rate']

    options = {'Adam': Adam, 'RMSprop': RMSprop}

    optimizer = options[name](learning_rate)

    return optimizer
