# Handwriting Recognition Tensorflow Problem
# We will be using the MNIST data set, a standard example for machine learning. This dataset has a collection of 70,000 handwriting samples of the 
# number 0-9. The challenge is to predict which number each handwritten image represents.

# In this example we dont need to use neural networks, for this is a relatively simple task. 

# In the dataset, each image is a 28 X 28 grayscale pixel image, so we are ble to treat each image as just a 1D array, or tensor, of 784 numbers.
# As long as we are consistent in how we flatten each image into an array, it will still work. In the later sections we work on preserving the 2D 
# structre of the data while training.

# First we import the MNIST dataset. It is built-in with Tensorflow

#%%
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#%%
# We will be using the Tensorflow session a few times. Ergo we will start it now.
sess = tf.InteractiveSession()

# MNIST provides 55,000 samples in the training dataset, 10,000 samples in the test dataset, and 5,000 samples in a validation dataset.
# Validation sets are inteded to be used for model selection.  We use validation data to select our model, train the model with the training set, 
# and then evaluate the model using the test set.
# It is important to evaluate the performance of our neural network using data it has never seen before. Used to test for accuracy.

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# The training data is therefore a tensor of shape [55,000, 784] - 55,000 instances of 784 numbers, represents each image.
# One hot is used to help determine the value the model believes it to be. The test data is encoded as 'one_hot' when we loaded it above. It can 
# be thought as the dirac function of which value is true. Example number 3 would be [0,0,0,1,0,0,0,0,0,0]
# The test data is a tensor of shape [55,000, 10] - 55,000 instances of 10 binary values, representing a given number 0-9.
#%%
# We will define a function that let's us visualize what the input data looks like.
def display_sample(num):
    # Print the one-hot array of this sample's label
    print(mnist.train.labels[num])
    # Print the label converted back to a number
    label = mnist.train.labels[num].argmax(axis=0)
    # Reshape the 768 values to a 28 X 28 image
    image = mnist.train.images[num].reshape([28,28])

    plt.title('Sample: %d Label: %d' %(num,label))
    plt.imshow(image,cmap =plt.get_cmap('gray_r'))
    plt.show()

#%%
display_sample(1234)

#%%
# Here we will visualize how the data is fed into the algorithm.
# An image of 500 sets of images.
images = mnist.train.images[0].reshape([1,784]) # Great way to add on to the image is using concatenate.

for i in range(1,500):
    images = np.concatenate((images,mnist.train.images[i].reshape([1,784])))

plt.imshow(images,cmap =plt.get_cmap('gray_r'))
plt.show()

#%%
# Here we start the setup for the artifical neural network.
# First we will start off by creating placeholders for the input images and the correct label for each.
# We can think of these as parameters that we build up our neural network model without knowledge of the actual data that will be fed into. 
# We need to construct it in such a way that our data will with in.
# So our 'input_images" placeholder will be set up to hold an array of values that consist of 784 floats, and our target_labels placehilder will be set
# up to hold an array of values that consist of 10 floats.
# While training, we will assign input_images to the training images and target_labels to the training data.
# While testing, we will use the test images and test labels instead.
input_images = tf.placeholder(tf.float32, shape=[None, 784])
target_labels = tf.placeholder(tf.float32, shape=[None,10])


#%%
hidden_nodes = 512

input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
input_biases =  tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

#%%
input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer,hidden_weights) + hidden_biases

#%%
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = digit_weights,labels = target_labels))

#%%
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)


#%%
correct_prediction = tf.equal(tf.argmax(digit_weights,1),tf.argmax(target_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#%%
tf.global_variables_initializer().run()

for x in range(2000):
    batch = mnist.train.next_batch(100)
    optimizer.run(feed_dict={input_images: batch[0], target_labels: batch[1]})
    if ((x+1) % 100 == 0):
        print("Training epoch " + str(x+1))
        print("Accuracy: " + str(accuracy.eval(feed_dict={input_images: mnist.test.images, target_labels:mnist.test.labels})))

#%%
for x in range(100):
    # Load a single test image and its label
    x_train = mnist.test.images[x,:].reshape(1,784)
    y_train = mnist.test.labels[x,:]
    # Convert the one-hot label to an integer
    label = y_train.argmax()
    # Get the classification from our neural network's digit_weights final layer, and convert it to an integer
    prediction = sess.run(digit_weights, feed_dict={input_images: x_train}).argmax()
    # If prediction does not match the correct label, display it
    if (prediction != label):
        plt.title('Prediction: %d Label %d' %(prediction,label))
        plt.imshow(x_train.reshape([28,28]), cmap= plt.get_cmap('gray_r'))
        plt.show()

#%%
