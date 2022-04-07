"""
CNN - cnn.py
============
A 1D convolutionar neural network for classification of times-series data. 
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from functools import partial
import tensorflow as tf

def get_model(num_classes=4, input_shape=(4800,1)):
    """Construct a 1D CNN model.

    Args:
        num_classes (int): The number of classes.
        input_shape (tuple): The shape of the input.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation=partial(tf.nn.leaky_relu, alpha=0.01), padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(filters=64, kernel_size=3, activation=partial(tf.nn.leaky_relu, alpha=0.01), padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
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
    