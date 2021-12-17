<h3>
Authors: Krzysztof Skwira & Tomasz Lemke
</h3>

Project consists of 4 cases which were resolved using neural network algorithms: \
<h4>#1 Banknote Dataset </h4> \
The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.
It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 1,372 observations with 4 input variables and 1 output variable. The variable names are as follows:

1. Variance of Wavelet Transformed image (continuous).
2. Skewness of Wavelet Transformed image (continuous).
3. Kurtosis of Wavelet Transformed image (continuous).
4. Entropy of image (continuous).
5. Class (0 for authentic, 1 for inauthentic). \

The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 50%. \
<b>This scenario was calculated using SVM with RBF kernel in Lab 4</b> \
<b>This time around we will use Neural Network algorithm to compare the results</b>

<h4>#2 CIFAR10 - animal/items recognition </h4> 

This dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Labels used for this dataset are: airplane, automobile, bird, cat, deer,
          dog, frog, horse, ship, truck. \
Based on the analysis pixel by pixel the algorythm classifies images to the above labels and in the end calculates the precision how well were they assigned.  

We used MLP (Multi-layer perceptron) Classifier with 2 hidden layers (256/128 neurons) with max of 100 iterations and using default (adam) solver, default (ReLU) activation function for the algorithm to learn on train data. 

Once the learning is completed the test data is being processed and accuracy is provided.

A confusion matrix is being plotted as a final step.
Logs, images and confusion matrix of the run are attached to the project. 

<h4>#3 A MNIST - fashion wear recognition </h4> 

This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.

Labels used for this dataset are: T-shirt/top, Trouser, Pullover, Dress, Coat,
          Sandal, Shirt, Sneaker, Bag, Ankle boot. \
Based on the analysis pixel by pixel the algorythm classifies images to the above labels and in the end calculates the precision how well were they assigned.  

We used 3 different MLP (Multi-layer perceptron) Classifiers to compare efficiency with max of 10 iterations for the algorithm to learn on train data. \
mlp1 = 1 hidden layer (100 neurons) \
mlp2 = 1 hidden layer (500 neurons) \
mlp3 = 1 hidden layer (700 neurons) \
All use default (adam) solver and default (ReLU) activation function.

Once the learning is completed the test data is being processed and accuracy is provided.

A confusion matrix is being plotted as a final step.
Images and logs of the run are attached to the project. 

<h4>#4 CIFAR100 - animal/items recognition </h4> 

Similar to CIFAR10 however it has 100 classes containing 600 images each. \
There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

For the full list of labels (Superclasses and Classes) please check the source website:
https://www.cs.toronto.edu/~kriz/cifar.html

We used MLP (Multi-layer perceptron) Classifier with 2 hidden layers (256/128 neurons) with max of 100 iterations and using default (adam) solver, default (ReLU) activation function for the algorithm to learn on train data. \
Once the learning is completed the test data is being processed and accuracy is provided.

A confusion matrix is being plotted as a final step.
Images and logs of the run are attached to the project. 

<h3>
Installation: 
</h3>

pip install numpy \
pip install pandas \
pip install itertools \
pip install matplotlib.pyplot \
pip install sklearn  
pip install keras \
pip install TensorFlow  
pip install opencv-python


<h3>
Reference:
</h3>


https://archive.ics.uci.edu/ml/datasets/banknote+authentication \
https://en.wikipedia.org/wiki/Variance \
https://en.wikipedia.org/wiki/Skewness \
https://en.wikipedia.org/wiki/Kurtosis \
https://keras.io/api/datasets/fashion_mnist/ \
https://www.cs.toronto.edu/~kriz/cifar.html \
https://scikit-learn.org/stable/modules/neural_networks_supervised.html