"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""

import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from sklearn import svm, model_selection, metrics
from sklearn.decomposition import PCA


# Reading the data from file
df = pd.read_csv('qualitative_bankruptcy.txt',
                 names=['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                        'Credibility', 'Competitiveness', 'Operating Risk', 'Class'])

bankruptcy_features = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                       'Credibility', 'Competitiveness', 'Operating Risk']

# Replacing the data with integer values
# Positive = 0
# Average = 1
# Negative = 2
replace_data_map = {'P': 1, 'A': 0, 'N': -1, 'B': 'Bankruptcy', 'NB': 'Non-Bankruptcy'}
df.replace(replace_data_map, inplace=True)

# Separating the features
X_bankruptcy = df.loc[:, bankruptcy_features].values

# Separating the target
Y_bankruptcy = df.loc[:, ['Class']].values.ravel()

# Splitting the data into training/testing data sets
# 40% of data reserved for testing and 60% data reserved for training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_bankruptcy, Y_bankruptcy,
                                                                    test_size=0.4, random_state=0)

# Building an SVC (Support Vector Classification) model using linear regression
svc = svm.SVC(kernel='linear', C=1, gamma=1).fit(X_train, Y_train)

# Validating the robustness of the model using K-Fold Cross validation technique
# We give the model, the entire data set and its real values, and the number of folds
scores_result = model_selection.cross_val_score(svc, X_bankruptcy, Y_bankruptcy, cv=5)

# Printing the accuracy of each fold and the mean of all 5 folds
print(f'Model accuracy scores: {scores_result}')
print(f'Model accuracy mean: {scores_result.mean():.2%}')

# Predicting the results using the test data
# and printing the fist 3 prediction values and accuracy of predictions
prediction_results = svc.predict(X_test)
print(f'The first company prediction is: {prediction_results[0]}')
print(f'The second company prediction is: {prediction_results[1]}')
print(f'The third company prediction is: {prediction_results[2]}')
print(f'Predictions accuracy score: {metrics.accuracy_score(Y_test, prediction_results):.2%}')

# Dimensionality Reduction using PCA (Principal Component Analysis)
# n_components = 2 means transforming into a 2-Dimensional dataset

pca = PCA(n_components=2, whiten=True).fit(X_bankruptcy)
X_pca = pca.transform(X_bankruptcy)

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
colors = it.cycle('rg')
target_names = ['Bankruptcy', 'Non-Bankruptcy']
plt.figure()
for t_name, c in zip(target_names, colors):
    plt.scatter(X_pca[Y_bankruptcy == t_name, 0], X_pca[Y_bankruptcy == t_name, 1], c=c, label=t_name)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA visualization')
plt.legend()
plt.show()
