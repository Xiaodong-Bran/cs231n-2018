import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #scores = np.zeros(num_train,num_class)
  scores = X.dot(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
        # compute Li
        fmax= np.max(scores[i])
        scores[i] -= fmax
        correct_class_score = scores[i,y[i]]
        M = np.exp(correct_class_score)/np.sum(np.exp(scores[i]))
        loss += -np.log(M)
        for j in range(num_class):
            N = np.exp(scores[i,j])/np.sum(np.exp(scores[i]))
            if j ==y[i]:
                dW[:,y[i]]+= (M-1)*X[i].T
            else:
                dW[:,j] += N*X[i].T                                        
  loss /= num_train
  loss += reg*np.sum(W*W)
  dW /=  num_train 
  dW += 2*reg*W                                         
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  num_class = W.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  temp_matrix = np.zeros(scores.shape)
 
  max_each_row = np.max(scores,axis=1).reshape(-1,1)
  scores -= max_each_row
  summation = np.sum(np.exp(scores),axis=1).reshape(-1,1)
  scores = np.exp(scores)
  scores = np.divide(scores,summation)
  temp_matrix[range(num_train),list(y)] =-1
  scores += temp_matrix
  dW = X.T.dot(scores) / num_train + 2*reg*W   
  log_summation = np.log(summation)
  vector = scores[range(num_train),list(y)].reshape(-1,1) 
  L = -vector+ log_summation 
  loss = np.sum(L)/num_train + reg*np.sum(W*W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

