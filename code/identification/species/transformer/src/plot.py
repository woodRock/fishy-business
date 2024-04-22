import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_attention_map(name, attention_weights, input_tokens, output_tokens):
    logger = logging.getLogger(__name__)
    # Plot attention weights as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest')

    # Set axis labels
    plt.xlabel('Mass-to-charge ratio')
    plt.ylabel('Mass-to-charge ratio')

    # Set axis ticks and labels
    plt.xticks(range(len(output_tokens)), output_tokens, rotation=45)
    plt.yticks(range(len(input_tokens)), input_tokens)

    # Show color bar
    plt.colorbar()

    # Add title
    plt.title(f'{name} Attention Map')
    file_path = f"figures/{name}_attention_map.png"
    plt.savefig(file_path)
    logger.info(f"Saving attention map to: {file_path}")
    # Show the plot (enable for interactive)
    # plt.show()

def plot_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
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

def plot_confusion_matrix(name, actual, predicted):
    actual = np.argmax(actual.cpu(), axis=0)
    predicted = np.argmax(predicted.cpu(), axis=0)
    cmatrix = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = [0, 1])

    cm_display.plot()
    plt.savefig(f"figures/{name}_confusion_matrix.png")
