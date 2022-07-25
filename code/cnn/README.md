# Convolutionary Neural Network 

# Requirements 

These are the external libraries required to run this code. 

```bash 
pip install numpy sklearn skfeature-chappers tensorflow matplotlib scipy
```

# Execution 

The code can be executed with the following command

```bash 
$ python3 -m cnn.Main
``` 

## Results 

This figure shows accuracy for the CNN model is given from the training and test set. 

![accuracy](./assets/accuracy.png)

This figure gives the confusion matrix for the training data.

![train confusion matrix](./assets/confusion_matrix_train.png)

This figure gives the confusion matrix for the test data. 

![accuracy](./assets/confusion_matrix_test.png)

The current model achieves 100% training and 98% test classification accuracy on the fish dataset. This is equivalent to the Support Vector Machine (SVM) results. 