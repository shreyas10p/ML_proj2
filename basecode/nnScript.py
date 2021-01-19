import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    exp = np.exp(-z)
    return  1/(1+exp)


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat') 
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_pre = np.zeros(shape=(50000, 784))
    train_len = 0
    valid_pre = np.zeros(shape=(10000, 784))
    validation_len = 0
    test_pre = np.zeros(shape=(10000, 784))
    test_len = 0
    train_l_pre = np.zeros(shape=(50000,))
    train_label_len = 0
    valid_l_pre = np.zeros(shape=(10000,))
    validation_label_len = 0
    test_l_pre = np.zeros(shape=(10000,))
    for key,val in mat.items():
        if "train" in key:
            label = key[-1]  
            val_range = range(len(val))
            val_perm = np.random.permutation(val_range)
            new_val_len = len(val)  - 1000  
            train_pre[train_len:train_len + new_val_len] = val[val_perm[1000:], :]
            train_len += new_val_len
            train_l_pre[train_label_len:train_label_len + new_val_len] = label
            train_label_len += new_val_len
            valid_pre[validation_len:validation_len + 1000] = val[val_perm[0:1000], :]
            validation_len += 1000
            valid_l_pre[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000
        elif "test" in key:
            label = key[-1]
            val_range = range(len(val))
            val_perm = np.random.permutation(val_range)
            test_l_pre[test_len:test_len + len(val) ] = label
            test_pre[test_len:test_len + len(val) ] = val[val_perm]
            test_len += len(val) 
    train_size = range(len(train_pre))
    validation_size = range(len(valid_pre))
    test_size = range(len(test_pre))
    train_perm = np.random.permutation(train_size)
    vali_perm = np.random.permutation(validation_size)
    test_perm = np.random.permutation(test_size)
    train_data = train_pre[train_perm]
    validation_data = valid_pre[vali_perm]
    test_data = test_pre[test_perm]
    train_data = np.double(train_data) / 255.0
    train_label = train_l_pre[train_perm]
    validation_data = np.double(validation_data) / 255.0
    validation_label = valid_l_pre[vali_perm]
    test_data = np.double(test_data) / 255.0
    test_label = test_l_pre[test_perm]

    # Feature selection
    #your code here
    global arr_remove
    arr_concat = np.concatenate((train_data, validation_data, test_data),axis=0)  
    arr_concat = np.array(arr_concat)
    checkData = np.all(arr_concat == arr_concat[0,:], axis = 0) 
    arr_remove = arr_concat[:,~checkData]
    train_data = arr_remove[0:len(train_data),:]
    validation_data = arr_remove[len(train_data): (len(train_data) + len(validation_data)),:]
    test_data = arr_remove[( len(train_data) + len(validation_data)): (len(train_data) + len(validation_data) + len(test_data)),:]

    print('preprocess done')
    # print(train_data)
    return train_data, train_label, validation_data, validation_label, test_data, test_label



def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    # print(training_label)
    newArr = []
    new_sigma = []
    zeros_arr= np.zeros((len(training_label), 10))

    zeros_arr[np.arange(len(training_label), dtype="int"), training_label.astype(int)] = 1
    # print("b",len(training_data))
    training_label = zeros_arr
    for row in training_data:
      # print(row)
      a = np.append(row,[1])
      newArr.append(a)
    training_data = np.array(newArr)
    # print(training_data)
    sigma_one = sigmoid(np.dot(training_data, w1.T))
    for row in sigma_one:
      a = np.append(row,[1])
      new_sigma.append(a)
    sigma_one = np.array(new_sigma)
    sigma_two = sigmoid(np.dot(sigma_one, w2.T))
    
    delta = sigma_two - training_label 
    
    grad_w2 = np.dot(delta.T, sigma_one)
    ele1 = ((1 - sigma_one) * sigma_one * (np.dot(delta, w2)))
    grad_w1 = np.delete(np.dot(ele1.T, training_data), n_hidden, 0)
  
    lamb_n = (lambdaval / (2 * len(training_data)))
    sum1 = np.sum(np.square(w1))
    sum2 = np.sum(np.square(w2))
    ele2 = (training_label * np.log(sigma_two) + (1 - training_label) * np.log(1 - sigma_two))
    ele3  =  lamb_n *( sum1 + sum2)
    obj_val = ((np.sum(-1 * ele2)) / len(training_data))
    obj_val = obj_val + ele3
    grad_w1 = (grad_w1 + (lambdaval * w1)) / len(training_data)
    grad_w2 = (grad_w2 + (lambdaval * w2)) / len(training_data)  
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    new_data = []
    sigma_one = []
    # Your code here
    for row in data:
      a = np.append(row,[1])
      new_data.append(a)
    data = np.array(new_data)
    sig_data = sigmoid( data.dot( w1.T))
    for row in sig_data:
      a = np.append(row,[1])
      sigma_one.append(a)
    
    sigma_one = np.array(sigma_one)
    sigma_two = sigmoid( sigma_one.dot( w2.T)) 

    labels = np.argmax( sigma_two, axis=1)
    return labels


"""**************Neural Network Script Starts here********************************"""
start_time = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# print("h",train_data)
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 60

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 40

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
print("total time",time.time() - start_time)
