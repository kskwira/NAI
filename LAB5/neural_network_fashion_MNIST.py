"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random


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
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Displaying 21 random images of X_train
ROW = 3
COLUMN = 7
plt.figure(figsize=(17, 8))
for i in range(ROW * COLUMN):
    temp = random.randint(0, len(X_train) + 1)
    image = X_train[temp]
    plt.subplot(ROW, COLUMN, i + 1)
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(labels[y_train[temp]])
    plt.tight_layout()
plt.show()

# Normalizing/Reshaping the data
X_train = X_train/255
X_test = X_test/255
# 28x28 grayscale = 28 * 28
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Building an MLP (Multi-layer perceptron) Classifier with 1 hidden layer (100 neurons)
# using default (adam) solver and default (ReLU) activation function
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10,
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
