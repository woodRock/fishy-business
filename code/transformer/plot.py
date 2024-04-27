import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_attention_map(name, attention_weights, y_axis, x_axis):
    """ Plot an attention map of an intermediary layer from the transformer.
    
    Args:
        name (str): the name for the layer
        attention_weights (np-array): the weights for the layer.
        y_axis (np-array): the y-axis 
        x_axis (np-array): the x-axis
    """
    logger = logging.getLogger(__name__)
    # Plot attention weights as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest')

    # Set axis labels
    plt.xlabel('Mass-to-charge ratio')
    plt.ylabel('Mass-to-charge ratio')

    # Set axis ticks and labels
    plt.xticks(range(len(x_axis)), x_axis, rotation=45)
    plt.yticks(range(len(y_axis)), y_axis)

    # Show color bar
    plt.colorbar()

    # Add title
    plt.title(f'{name} Attention Map')
    file_path = f"figures/{name}_attention_map.png"
    plt.savefig(file_path)
    logger.info(f"Saving attention map to: {file_path}")
    # Show the plot (enable for interactive)
    # plt.show()
    # Too many open figures
    # Source: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    plt.clf()
    plt.close()

def plot_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    """ Plot the accuracy and loss curve for the training process.

    This method takes the output from the training process and turns it into a graph.
    
    Args: 
        train_losses (np-array): the array for training losses.
        val_losses (np-array): the array for validation losses.
        train_accuracies (np-array): the array for training accuracies.
        val_accuracies (np-array): the array for validation accuracies.
    """
    logger = logging.getLogger(__name__)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.plot(train_accuracies, label="train_acc")
    plt.plot(val_accuracies, label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    file_path = "figures/model_accuracy.png"
    plt.savefig(file_path)
    logger.info(f"Saving attention map to: {file_path}")
    # Show the plot (enable for interactive)
    # plt.show()
    # Too many open figures
    # Source: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    plt.clf()
    plt.close()
    

def plot_confusion_matrix(dataset, name, actual, predicted):
    """ Plots a confusion matrix for a dataset.
    
    Args: 
        dataset (np-array): train, validation or test dataset.
        name (str): the name of the datasset for titles.
        actual (np-array): the expected values for y labels.
        predicted (np-array): the predicted values for y labels.
    """
    logger = logging.getLogger(__name__)
    cmatrix = confusion_matrix(actual, predicted)
    labels = [] 
    if dataset == "species":
        labels = ["Hoki", "Mackerel"]
    elif dataset == "part":
        labels = ["Fillet", "Heads", "Livers", "Skins", "Guts", "Frames"]
    # Disable the grid on the confusion matrix 
    # Source: https://stackoverflow.com/questions/53574918/how-to-get-rid-of-white-lines-in-confusion-matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=labels)
    cm_display.plot()
    plt.grid(False)
    file_path = f"figures/{name}_confusion_matrix.png"
    logger.info(f"Saving cofusion matrix map to: {file_path}")
    plt.savefig(file_path)
    # Too many open figures
    # Source: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    plt.clf()
    plt.close()
