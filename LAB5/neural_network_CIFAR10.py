"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


# Defining the Confusion Matrix
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Loading data from Karas dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Displaying the first 21 images of X_train
fig, axes = plt.subplots(ncols=7, nrows=3, figsize=(17, 8))
index = 0
for i in range(3):
    for j in range(7):
        axes[i, j].set_title(labels[y_train[index][0]])
        axes[i, j].imshow(X_train[index])
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)
        index += 1
plt.show()

# Normalizing/Reshaping the data
X_train = X_train/255
X_test = X_test/255
# 32x32 RGB = 28 * 28 * 3
X_train = X_train.reshape(-1, 3072)
X_test = X_test.reshape(-1, 3072)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Building an MLP (Multi-layer perceptron) Classifier with 2 hidden layers (256/128 neurons)
# using default (adam) solver and default (ReLU) activation function
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=10,
                    verbose=10, random_state=1)

# This example won't converge because of time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

# Printing the training set & test set scores
print(f'Training set score: {mlp.score(X_train, y_train):.2%}')
print(f'Test set score: {mlp.score(X_test, y_test):.2%}')

# Printing the Confusion Matrix and scores
prediction = mlp.predict(X_test)
f1_score = metrics.f1_score(y_test, prediction, average="weighted")
accuracy_score = metrics.accuracy_score(y_test, prediction)
confusion_matrix = metrics.confusion_matrix(y_test, prediction)
print("-------------------SVM Report-----------------")
print(f"F1 score: {f1_score:.2%}")
print(f"Accuracy score: {accuracy_score:.2%}")
print("Confusion matrix: \n", confusion_matrix)
print('Plotting confusion matrix')
plt.figure()
plot_confusion_matrix(confusion_matrix, labels)
plt.show()
print(metrics.classification_report(y_test, prediction))
