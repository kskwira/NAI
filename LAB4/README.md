<h3>
Authors: Krzysztof Skwira & Tomasz Lemke
</h3>

Project consists of 2 cases which both were resolved using SVM algorithms: \
<h4>Banknote Dataset </h4> \
The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.
It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 1,372 observations with 4 input variables and 1 output variable. The variable names are as follows:

1. Variance of Wavelet Transformed image (continuous).
2. Skewness of Wavelet Transformed image (continuous).
3. Kurtosis of Wavelet Transformed image (continuous).
4. Entropy of image (continuous).
5. Class (0 for authentic, 1 for inauthentic). \

The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 50%.

<b>For the calculations we used SVM with RBF kernel</b>

<h4>Qualitative Bankruptcy Data Set</h4> \
Predict the Bankruptcy from Qualitative parameters from experts for Companies in India.

The parameters which we used for collecting the dataset is referred from the paper 'The discovery of expertise™ decision rules from qualitative bankruptcy data using genetic algorithms' by Myoung-Jong Kim*, Ingoo Han.


Attribute Information: (P=Positive,A-Average,N-negative,B-Bankruptcy,NB-Non-Bankruptcy)

1. Industrial Risk: {P,A,N}
2. Management Risk: {P,A,N}
3. Financial Flexibility: {P,A,N}
4. Credibility: {P,A,N}
5. Competitiveness: {P,A,N}
6. Operating Risk: {P,A,N}
7. Class: {B,NB}

<b>For the calculations we used SVM with Linear kernel</b>

<h3>
Installation: 
</h3>

pip install numpy \
pip install pandas \
pip install itertools \
pip install matplotlib.pyplot \
pip install sklearn


<h3>
Reference:
</h3>


https://en.wikipedia.org/wiki/Euclidean_distance \
https://scikit-learn.org/stable/modules/svm.html#svm-classification \
https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html \
https://en.wikipedia.org/wiki/Variance \
https://en.wikipedia.org/wiki/Skewness \
https://en.wikipedia.org/wiki/Kurtosis