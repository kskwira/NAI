"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import warnings
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import random


# Loading data from Karas dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

# Building 3 different MLP (Multi-layer perceptron) Classifiers to compare efficiency
# mlp1 = 1 hidden layer (100 neurons)
# mlp2 = 1 hidden layer (500 neurons)
# mlp3 = 1 hidden layer (700 neurons)
# all use default (adam) solver and default (ReLU) activation function
mlp1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10,
                     verbose=10, random_state=1)
mlp2 = MLPClassifier(hidden_layer_sizes=(500,), max_iter=10,
                     verbose=10, random_state=1)
mlp3 = MLPClassifier(hidden_layer_sizes=(700,), max_iter=10,
                     verbose=10, random_state=1)

# This example won't converge because of time constraints, so we catch and ignore the warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp1.fit(X_train, y_train)
    mlp2.fit(X_train, y_train)
    mlp3.fit(X_train, y_train)

# Printing the training set & test set scores
print("-------------------SET SCORES-----------------")
print('MLP1 SCORES: 1 hidden layer (100 neurons)')
print(f'Training set score: {mlp1.score(X_train, y_train):.2%}')
print(f'Test set score: {mlp1.score(X_test, y_test):.2%}')
print('MLP2 SCORES: 1 hidden layer (500 neurons)')
print(f'Training set score: {mlp2.score(X_train, y_train):.2%}')
print(f'Test set score: {mlp2.score(X_test, y_test):.2%}')
print('MLP3 SCORES: 1 hidden layer (700 neurons)')
print(f'Training set score: {mlp3.score(X_train, y_train):.2%}')
print(f'Test set score: {mlp3.score(X_test, y_test):.2%}')
