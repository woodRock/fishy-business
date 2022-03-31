"""
Main - main.py 
==============

This is the main routine for the CNN model. 
It trains a 1-dimensional convolutionary neural network on the fish oil data. 
We perform the classification task using this model - it predicts the fish species or part. 
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from .data import load, prepare, normalize, encode_labels
from .cnn import get_model  
from .plot import show_confusion_matrix, plot_loss, plot_accuracy

# Hyperparameters
batch_size = 32
epochs = 30

# Dataset
folder = "data/matlab/"
datasets = ["Fish.mat","Part.mat"]
dataset = datasets[0]
print("Chosen: %s" % dataset)

mat = load(dataset,folder=folder)
X,y = prepare(mat)

history = []

skf = StratifiedKFold(n_splits=10, random_state=1234, shuffle=True)
for train, test in skf.split(X, y):
    X_train, X_test = normalize(X[train], X[test])
    X_train = np.expand_dims(X_train, axis=-1) # Batch axis (source: https://bit.ly/3yAF1J7)
    X_test = np.expand_dims(X_test, axis=-1) # Batch axis (source: https://bit.ly/3yAF1J7)
    input_shape = (X_train.shape[1], X_train.shape[2])
    # print(f"Input shape: {input_shape}")
    y_train, y_test, le = encode_labels(y[train], y[test])
    class_names = le.inverse_transform(np.unique(y_train)).tolist()
    # print(class_names)
    num_classes = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes) # Convert labels to one-hot encoding.
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes) # Convert labels to one-hot encoding.
    # X_train, X_val, y_train, y_val = train_test_split(X,y)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(1000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_test,y_test)).shuffle(1000).batch(batch_size)
    model = get_model(num_classes=num_classes)
    history.append(model.fit(train_ds, validation_data=val_ds, epochs=epochs))

print(f"History: {history[0].history}")

for i in range(epochs):
    avg_loss = [] 
    avg_val_loss = []
    avg_acc = []
    avg_val_acc = []
    
    for j in range(len(history)):
        avg_loss.append(history[j].history['loss'][i])
        avg_val_loss.append(history[j].history['val_loss'][i])
        avg_acc.append(history[j].history['accuracy'][i])
        avg_val_acc.append(history[j].history['val_accuracy'][i])
    
    avg_loss, std_loss = np.mean(avg_loss), np.std(avg_loss)
    avg_val_loss, std_val_loss = np.mean(avg_val_loss), np.std(avg_val_loss)
    avg_acc, std_acc = np.mean(avg_acc), np.std(avg_acc)
    avg_val_acc, std_val_acc = np.mean(avg_val_acc), np.std(avg_val_acc)

    print (f"Epoch: {i}, loss: {avg_loss:4f} +/- {std_loss:4f}, val_loss: {avg_val_loss:4f} +/- {std_val_loss:4f}, acc: {avg_acc:4f} +/- {std_acc:4f}, val_acc: {avg_val_acc:4f} +/- {std_val_acc:4f}")

# plot_accuracy(history)
# plot_loss(history)
# show_confusion_matrix(X_val, y_val, model, labels=class_names)