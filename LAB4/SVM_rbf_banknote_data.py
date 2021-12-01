"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, model_selection, metrics
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

# Building an SVC (Support Vector Classification) model using radial basis function (RBF)
svc = svm.SVC(kernel='rbf', C=1, gamma=1).fit(X_train, Y_train)

# Validating the robustness of the model using K-Fold Cross validation technique
# We give the model, the entire data set and its real values, and the number of folds
scores_result = model_selection.cross_val_score(svc, X_banknote, Y_banknote, cv=5)

# Printing the accuracy of each fold and the mean of all 5 folds
print(f'Model accuracy scores: {scores_result}')
print(f'Model accuracy mean: {scores_result.mean():.2%}')

# Predicting the results using the test data
# and printing the fist 3 prediction values and accuracy of predictions
prediction_results = svc.predict(X_test)
print(f'The first given banknote is: {prediction_results[0]}')
print(f'The second given banknote is: {prediction_results[1]}')
print(f'The third given banknote is: {prediction_results[2]}')
print(f'Predictions accuracy score: {metrics.accuracy_score(Y_test, prediction_results):.2%}')

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
