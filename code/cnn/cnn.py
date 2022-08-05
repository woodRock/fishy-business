"""
CNN - cnn.py
============
A 1D convolutionar neural network for classification of times-series data. 
These are models designed through the 'black magic' technique of domain expertise and trial and error.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from functools import partial
import tensorflow as tf


def get_model(num_classes=4, input_shape=(4800, 1), dataset='fish'):
    """Construct a 1D CNN model.

    This method returns the manually tuned model for each dataset. 

    Args:
        num_classes (int): The number of classes.
        input_shape (tuple): The shape of the input.
        dataset (str): The dataset to use. Determines the manually tuned model to use.
    """

    if dataset[0:4] == 'Fish':
        # Fish species dataset model.
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation=partial(
                tf.nn.leaky_relu, alpha=0.01), padding='same', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Conv1D(filters=64, kernel_size=3, activation=partial(
                tf.nn.leaky_relu, alpha=0.01), padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Flatten(),
            Dense(64, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
            Dense(10, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
            Dense(num_classes, activation='softmax')
        ])
    else:
        # Fish part dataset model.
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation=partial(
                tf.nn.leaky_relu, alpha=0.01), padding='same', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.9),
            Flatten(),
            Dense(64, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
            Dense(10, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
            Dense(num_classes, activation='softmax')
        ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
