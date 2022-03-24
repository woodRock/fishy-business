"""
Main - main.py 
==============

This is the main routine for the CNN model. 
It trains a 1-dimensional convolutionary neural network on the fish oil data. 
We perform the classification task using this model.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .data import load, prepare, normalize, encode_labels
from .cnn import get_model  
from .plot import show_confusion_matrix, plot_loss, plot_accuracy

folder = "data/matlab/"
datasets = ["Fish.mat","Part.mat"]
dataset = datasets[0]
print("Chosen: %s" % dataset)
mat = load(dataset,folder=folder)

X,y = prepare(mat)
X,_ = normalize(X,X)
y, le = encode_labels(y)

class_names = le.inverse_transform(np.unique(y)).tolist()
print(class_names)
num_classes = len(np.unique(y))

# Convert labels to one-hot encoding.
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Batch axis (source: https://bit.ly/3yAF1J7)
X = np.expand_dims(X, axis=-1)

input_shape = (X.shape[1],X.shape[2])
batch_size = 32

print(f"Input shape: {input_shape}")

X_train, X_val, y_train, y_val = train_test_split(X,y)

train_ds = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(1000).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((X_val,y_val)).shuffle(1000).batch(batch_size)

model = get_model(num_classes=num_classes)
history = model.fit(train_ds, validation_data=val_ds, epochs=20)

plot_accuracy(history)
plot_loss(history)
show_confusion_matrix(X_val, y_val, model, labels=class_names)