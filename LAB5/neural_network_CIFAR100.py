"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import warnings
import matplotlib.pyplot as plt
from keras.datasets import cifar100
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


# Loading data from Karas dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")
labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

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
X_train = X_train / 255
X_test = X_test / 255
# 32x32 RGB = 32 * 32 * 3
X_train = X_train.reshape(-1, 3072)
X_test = X_test.reshape(-1, 3072)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Building an MLP (Multi-layer perceptron) Classifier with 2 hidden layers (256/128 neurons)
# using default (adam) solver and default (ReLU) activation function
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100,
                    verbose=10, random_state=1)

# This example won't converge because of time constraints, so we catch and ignore the warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

# Printing the training set & test set scores
print(f'Training set score: {mlp.score(X_train, y_train):.2%}')
print(f'Test set score: {mlp.score(X_test, y_test):.2%}')
