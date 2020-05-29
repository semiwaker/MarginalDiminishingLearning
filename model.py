import os.path as path

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import option


def makeBaselineModel(params):
    return keras.Sequential(
        (
            layers.Input([32, 32, 3]),
            layers.Conv2D(32, 3, padding="same",
                          strides=(2, 2)),  # (32, 16, 16)
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.MaxPool2D(),  # (32, 8, 8)
            layers.Conv2D(64, 3, padding="same"),  # (64, 8, 8)
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.MaxPool2D(),  # (64, 4, 4)
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10),
            layers.Softmax()
        ),
        "Baseline"
    )


class SplitModel(keras.module):
    def __init__(self, params):
        super(SplitModel, self).__init__()
        self.conv = keras.Sequential(
            layers.Input([32, 32, 3]),
            layers.Conv2D(32, 3, padding="same",
                          strides=(2, 2)),  # (32, 16, 16)
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.MaxPool2D(),  # (32, 8, 8)
            layers.Conv2D(64, 3, padding="same"),  # (64, 8, 8)
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.MaxPool2D(),  # (64, 4, 4)
            layers.Flatten(),
            layers.Dense(1024, activation="relu")
        )
        self.predict = keras.Sequential(
            layers.Dropout(0.2),
            layers.Dense(10),
            layers.Softmax()
        )

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        y = self.predict(x, training=training)
        return x, y


class CateAcc(keras.metrics.Metric):
    def __init__(self, name="CateAcc", **kwargs):
        super(CateAcc, self).__init__(name=name, **kwargs)
        self.cate_sum = self.add_weight(
            name="CateSum", shape=[10], initializer="zeros")
        self.total = self.add_weight(
            name="total", shape=[10], initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = y_true * y_pred
        self.cate_sum.assign_add(tf.reduce_sum(values, axis=0))
        self.total.assign_add(tf.reduce_sum(y_true, axis=0))

    def reset_states(self):
        self.cate_sum.assign(tf.zeros([10]))
        self.total.assign(tf.zeros([10]))

    def result(self):
        return self.cate_sum / self.total
