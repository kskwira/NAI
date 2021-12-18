"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Reading the data from file
df = pd.read_csv('banknote_authentication.txt', names=['Variance of Wavelet', 'Skewness of Wavelet',
                                                       'Kurtosis of Wavelet', 'Entropy', 'Class'])

banknote_features = ['Variance of Wavelet', 'Skewness of Wavelet', 'Kurtosis of Wavelet', 'Entropy']

# Encoding the labels
banknote_labels = ['Authentic', 'Fake']
encoder = preprocessing.LabelEncoder()
encoder.fit(banknote_labels)

# Separating the features
X_banknote = df.loc[:, banknote_features].values

# Separating the target
Y_banknote = df.loc[:, ['Class']].values.ravel()

# Encoding the target
Y_banknote = encoder.inverse_transform(Y_banknote)

# Standardizing the features
X_banknote = StandardScaler().fit_transform(X_banknote)

# Splitting the data into training/testing data sets
# 40% of data reserved for testing and 60% data reserved for training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_banknote, Y_banknote,
                                                                    test_size=0.4, random_state=0)

# Building an MLP (Multi-layer perceptron) Classifier with 1 hidden layer (50 neurons)
# using sgd (Stochastic Gradient Descent) solver and default (ReLU) activation function
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, solver='sgd',
                    verbose=10, random_state=1, learning_rate_init=0.1)
mlp.fit(X_train, Y_train)

# Printing the training set & test set scores
print(f'Training set score: {mlp.score(X_train, Y_train):.2%}')
print(f'Test set score: {mlp.score(X_test, Y_test):.2%}')

# Dimensionality Reduction using PCA (Principal Component Analysis)
# n_components = 2 means transforming into a 2-Dimensional dataset
pca = PCA(n_components=2, whiten=True).fit(X_banknote)
X_pca = pca.transform(X_banknote)

"""
The explained variance tells you how much information (variance) can be attributed
to each of the principal components. This is important as while you can convert
4 dimensional space to 2 dimensional space, you lose some of the variance (information)
when you do this
"""

print(f'Explained variance ratio for component 1: {pca.explained_variance_ratio_[0]:.2%}')
print(f'Explained variance ratio for component 2: {pca.explained_variance_ratio_[1]:.2%}')
print(f'Preserved variance sum: {sum(pca.explained_variance_ratio_):.2%}')

# Printing scatter plot to view classification of the simplified dataset
colors = it.cycle('gr')
target_names = banknote_labels
plt.figure()
for t_name, c in zip(target_names, colors):
    plt.scatter(X_pca[Y_banknote == t_name, 0], X_pca[Y_banknote == t_name, 1], c=c, label=t_name)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA visualization')
plt.legend()
plt.show()
