"""
Plot - ploy.py
==============

This file contains the functions for plotting the results of the model.
It can display loss, accuracy and confusion matrices for the classification task.
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def show_confusion_matrix(X_test, y_test, model, labels = ['BCO', 'GUR', 'SNA', 'TAR']):
    """Shows the confusion matrix for the test data. 

    Args:
        X_test ([[int]]): The test data.
        y_test ([[int]]): Onehot encoded labels for the test data.
        model (keras.model): The model to use for the prediction.
    """
    predictions = model.predict(x=X_test, steps=len(X_test),verbose=0)
    cm = confusion_matrix(
        y_true = np.argmax(y_test, axis=-1),
        y_pred = np.argmax(predictions, axis=-1)
    )
    cm_plot_labels = labels
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

def plot_confusion_matrix(cm, classes, title='Confsion matrix', cmap = plt.cm.Blues):
    """Plot a confusion matrix for a classification task.

    Args:
        cm (np.array): The confusion matrix.
        classes ([str]): The labels of the classes.
        title (str): The title of the plot.
        cmap (plt.cm): The color map to use.
    """
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2. 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else "black")
    plt.tight_layout() 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#     plt.savefig("../report/assets/confusion_matrix.png")
    plt.savefig('cnn/assets/confusion_matrix.png')
    plt.show()

def plot_accuracy(history):
    """Plots the accuracy of the model.
    
    Args:
        history (keras.history): The history of the model.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cnn/assets/accuracy.png')
    plt.show()

def plot_loss(history):
    """Plots the loss of the model.
    
    Args:
        history (keras.history): The history of the model.
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cnn/assets/loss.png')
    plt.show()