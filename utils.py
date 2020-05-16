import time
import tensorflow as tf


class Timer:
    def __init__(self):
        self.reset()

    def __call__(self):
        delta = int(time.time() - self.start_time)
        h = delta//3600
        m = (delta % 3600) // 60
        s = delta % 60
        result = ""
        if h > 0:
            result += f"{h} hours "
        if h > 0 or m > 0:
            result += f"{m} minutes "
        result += f"{s} seconds"
        return result

    def reset(self):
        self.start_time = time.time()


def selectDevice(device):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and device >= 0:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpus[device], True)
            tf.config.experimental.set_visible_device(gpus[device])
            print(gpus[device])
            return gpus[device]
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if device < 0:
        return '/CPU:0'


def getDevice():
    return tf.config.experimental.list_physical_devices('GPU')
