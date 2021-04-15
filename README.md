# Handwritten-Digit-Recognition


Digit Recognition is a noteworthy and important issue. As the manually written digits are not of a similar size, thickness, position, and direction, in this manner, various difficulties must be considered to determine the issue of handwritten digit recognition. The uniqueness and assortment in the composition styles of various individuals additionally influence the example and presence of the digits. 
The task of handwritten digit recognition, using a classifier, is to predict the digits written in the canvas by comparing them with MNIST Dataset.

MNIST is the most broadly utilized standard for handwritten digit recognition. MNIST dataset has been commonly used as a standard for testing classification algorithms in handwritten digit recognition frameworks.
The initial step to be carried out is to place the dataset, which can be effectively done through the Keras programming interface. The images in the MNIST dataset comprise 28x28 pixel images and there is a total of 70000 images that are divided into 60000 which is given for training whereas 10000 is given to test data. Images are 2D matrix where each pixel is represented between [0,255] here 0 means black,255 means white.
The image is then normalized in the range of 0 to 1 and resized to add an extra dimension for the kernel. It describes the Data flow diagram of the proposed system model. The user draws the digit in the canvas which is then detected using the MNIST dataset. The input images are pre-processed. Using the CNN classifier, the recognized digits' accuracy is compared, and the result is obtained. The results obtained are displayed along with the accuracy. 
The model is also passed to the confusion matrix, it checks that which true value was predicted wrong by the model.

Utilizing these deep learning techniques, a high amount of accuracy can be obtained. Compared to other research methods, this method focuses on which classifier works better by improving the accuracy of classification models by more than 99%. 
Using Keras as backend and TensorFlow as the software, a CNN model can give an accuracy of about 98.7% while Decision and Random Forest have less accuracy compared to it
