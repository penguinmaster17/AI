
# coding: utf-8

# In[1]:

import os
import urllib
from urllib import request
import gzip
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def cached_download(filename):
    """
    Only download data files if necessary.
    """
    try:
        os.stat("./")
    except:
        os.mkdir("./")
    #
    filepath = os.path.join("./", filename)
    #
    if not os.path.isfile(filepath):
        filepath, headers = urllib.request.urlretrieve(URLBASE + filename, filepath)
    #
    return filepath


# In[2]:

IMAGEW = 28
IMAGEH = 28
LABELS = 10


# In[3]:

def unpack_files(imagefile, labelsfile, count):
    with gzip.open(imagefile) as f:
        f.read(16)
        buf = f.read(IMAGEW * IMAGEH * count)
        images = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)
        images = images.reshape(count, IMAGEW, IMAGEH)

    with gzip.open(labelsfile) as f:
        f.read(8)
        labels = np.frombuffer(f.read(1 * count), dtype = np.uint8).astype(np.int64)
    
    return (images, labels)


# In[4]:

URLBASE = 'http://yann.lecun.com/exdb/mnist/'

images_pathname = cached_download('train-images-idx3-ubyte.gz')
labels_pathname = cached_download('train-labels-idx1-ubyte.gz')

images_3d, labels = unpack_files(images_pathname, labels_pathname, 60000)


# In[5]:

print(images_3d.shape, labels.shape)


# In[6]:

imgmatrix = np.vstack([np.hstack([images_3d[random.randrange(len(labels)), :, :] for i in range(10)]) for j in range(10)])


# In[7]:

plt.imshow(imgmatrix, interpolation = 'nearest')
plt.axis('off')
plt.show()


# In[8]:

# -1 tells python that we want the dimension to be 2D but we don't know how long the second axis is going to be
images = images_3d.reshape(len(labels), -1)


# In[9]:

print(images.shape)


# In[10]:

X, y = images, labels


# In[11]:

idim = IMAGEW * IMAGEH
odim = LABELS

hdim = 25
epsilon = 0.00001
LAMBDA = 0.01
NBATCH = 2000
N = len(labels)


# In[12]:

def soft_max(scores):
    """
    Convert scores to probabilities.
    """
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

def predict(model, x):
    """
    Generate a prediction via forward propagation.
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Input values to hidden layer.
    z1 = x.dot(W1) + b1
    # Activation of hidden layer.
    a1 = np.tanh(z1)
    # Input to ouput layer.
    z2 = a1.dot(W2) + b2
    
    return soft_max(z2)

def predict_class(model, x):
    return np.argmax(predict(model, x), axis = 1)


# In[13]:

def loss(model):
    """
    Evaluate cross-entropy loss.
    """
    W1, W2 = model['W1'], model['W2']
    #
    probs = predict(model, X)
    #
    L = np.sum(-np.log(probs[range(N), y]))
    # Add regularization.
    L += LAMBDA / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    #
    return L / N


# In[14]:

def build_model(hdim, passes = 20000, verbose = False):
    """
    Train model using backward propagation and batch gradient descent.
    
    :param hdim: Number of hidden nodes.
    :param passes: Number of iterations.
    :param verbose: Whether or not to produce status information.
    :return: data frame.
    """
    np.random.seed(0)
    #
    # Initialise the weights and biases.
    #
    W1 = np.random.randn(idim, hdim) / np.sqrt(idim)
    b1 = np.zeros((1, hdim))
    W2 = np.random.randn(hdim, odim) / np.sqrt(hdim)
    b2 = np.zeros((1, odim))
    #
    # W1 is a (idim, hdim) array
    # b1 is a (1, hdim) array
    # W2 is a (hdim, odim) array
    # b2 is a (1, odim) array
    #
    # X is a (N, idim) array
    # y is a (N, 1) array
    
    for i in range(0, passes):
        index = random.sample(range(len(labels)), NBATCH)
        X = images[index,]
        y = labels[index]
        
        # Generate predictions with current model via forward propagation.
        #
        # (N, idim) x (idim, hdim) -> (N, hdim)
        #
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        #
        # z1 is input to the hidden layer.
        # a1 is output from the hidden layer (after applying the activation function).
        #
        # (N, hdim) x (hdim, odim) -> (N, odim)
        #
        z2 = a1.dot(W2) + b2
        #
        # z2 is input to the output layer.
        #
        delta = soft_max(z2)
        #
        # delta is a (   N, odim) array

        # Backward propagation.
        #
        # Takes advantage of d/dx tanh(x) = 1 - tanh(x)^2.
        #
        delta[range(NBATCH), y] -= 1
        #
        # (idim, N) x (N, odim) -> (idim, odim)
        #
        # Back propagation below differentiates the output in order to find 
        # the gradient which has led the output there. This will give
        # the right direction to which the model should take its step
        
        dW2 = (a1.T).dot(delta)
        db2 = np.sum(delta, axis = 0, keepdims = True)
        delta2 = delta.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Update model parameters, including regularisation.
        # You want to move a fraction of the gradient, which is determined by the learning rate
        
        W1 += -epsilon * (dW1 + LAMBDA * W1)
        b1 += -epsilon * db1
        W2 += -epsilon * (dW2 + LAMBDA * W2)
        b2 += -epsilon * db2
        
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        if verbose and i % 1000 == 0:
          print("iteration %i: Loss = %f" %(i, loss(model)))
    
    return model


# In[15]:

model = build_model(hdim, verbose = True)


# In[16]:

correct = 0
incorrect = 0
for i in range(len(labels)):
    prediction = predict_class(model, images[i,])
    actual = labels[i]
    if actual == prediction:
        correct += 1
    else:
        plt.imshow(images_3d[i, :, :], interpolation = 'nearest')
        plt.axis('off')
        plt.figure(figsize=(2,2))
        plt.show()
        print(predict_class(model, images[i, ]))
accuracy = correct / len(labels)
print(incorrect)


# In[17]:

print(accuracy)


# In[ ]:

# for k in range(len(labels)):  
#     plt.imshow(images_3d[k, :, :], interpolation = 'nearest')
#     plt.axis('off')
#     plt.figure(figsize=(2,2))
#     plt.show()
#     print(predict_class(model, images[k,]))


# In[ ]:

test_images_pathname = cached_download('t10k-images-idx3-ubyte.gz')
test_labels_pathname = cached_download('t10k-images-idx3-ubyte.gz')


# In[ ]:



