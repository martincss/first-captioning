import os
import tensorflow as tf
from params import MEMORY_LIMIT, GPU_NUMBER
#
# def running_on_cluster():
#
#     hostname = os.uname()[1]
#
#     return hostname == 'nabucodonosor2'


def enable_gpu_memory_growth():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            if GPU_NUMBER:
                tf.config.experimental.set_visible_devices(gpus[GPU_NUMBER], 'GPU')
            for gpu in gpus:
                if MEMORY_LIMIT is None:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(
                    gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
