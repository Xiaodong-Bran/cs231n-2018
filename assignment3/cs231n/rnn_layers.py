from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    #pass
    x1 = x.dot(Wx)
    x2 = prev_h.dot(Wh)
    x3 = x1 + x2 + b
    next_h = np.tanh(x3)
    cache=(b,x,prev_h,Wx,Wh,next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    b,x,prev_h,Wx,Wh,next_h = cache
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    #pass
    dnext_hdx3 =  (1 - np.square(next_h))  ##size(tanh(x3))=size(x3)=size(next_h)= ((N, H))
    dtanh = dnext_hdx3 * dnext_h  #@@@@@ key difference is here. Just find out what happens.
    ###############################@@@ this is the element-wise multiply. It is okay.
    ############EXCEPT this, all otheres should observe the dot 
    dx3db = 1 # it is wrong.
    ############
    dx3dx = Wx.T
    dx3dprev_h = Wh.T
    dx3dWx = x.T
    dx3dWh = prev_h
    
    # dx = dnext_h* dnext_hdx3*dx3dx @@@@@ From the matrix to vector derivative.Just find out what happens.
    db = np.sum(dtanh,axis=0)
    # dprev_h = dnext_h*dnext_hdx3*dx3dprev_h
    # @@@@@@ when use dot product, we should pay attention to the shape of the matrix
    # can not use element-wize multiply. Should use dot multiply
    # dWx = dnext_h*dnext_hdx3*dx3dWx
    # dWh = dnext_h*dnext_hdx3*dx3dWh
    # db = dnext_h*dnext_hdx3*dx3db
    dprev_h = np.dot(dtanh,dx3dprev_h)
    dWx = np.dot(dx3dWx,dtanh)
    dx = np.dot(dtanh,Wx.T)
    dWh = np.dot(prev_h.T,dtanh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # pass
    # @@@@ how to feed this data structure (N,T,D) into the original function for 
    # data structure of (N,D)
    ### the understanding and solution is that we can extract / slice at the dimension T
    # for example (:,1,:) will return an array of the shape (N,D)
    N,T,D  = x.shape
    N, H = h0.shape
    ## @@@ pre-pare the space for h
    h = np.zeros((N,T,H))
    cache = []
    prev_h = h0
    #h[:,0,:]=h0 @@@ h0 should not be included in the matrix
    for i in range(0,T):
            curr_x = x[:,i,:]
            ### @@@ output with prev_h which can omit the assignment prev_h = next_h
            prev_h,cache_temp = rnn_step_forward(curr_x, prev_h, Wx, Wh, b)
            h[:,i,:] = prev_h
            cache.append(cache_temp)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    #pass
    
    #cache=(x3,x,prev_h,Wx,Wh,next_h)
    b,x,prev_h,Wx,Wh,next_h = cache[0]
    #Wx, Wh, b, x, prev_h, next_h = cache[0]
    N, T, H = dh.shape
    D, H = Wx.shape
    
    # save space 
    # Initialise gradients.
    dx = np.zeros([N, T, D])
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h_temp = np.zeros_like(prev_h)
    
    for t_step in np.arange(T-1,-1,-1):
        
        cache_temp = cache[t_step]
        #@@@ dnext_h should also include the previous derivative (look at the computation graph)
        dnext_h = dh[:,t_step,:] + dprev_h_temp
        #dx_temp, dprev_h_temp, dWx_temp, dWh_temp, db_temp=rnn_step_backward(dnext_h, cache):
        #dx[:,t_step,:] = dx_temp
        # can be mixed into one line
        dx[:,t_step,:],dprev_h_temp, dWx_temp, dWh_temp, db_temp=rnn_step_backward(dnext_h, cache_temp)
        # see the summation from the equation of mathematics
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
    dh0 = dprev_h_temp
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    #pass
    if isinstance(x, int):
        N,T,=1,1
    else:
        T = x.shape
        N =1 
    V,D = W.shape
    
    # initialization 
    #out = np.zeros([N,T,D])
    #temp = np.zeros([T,D])
    
    # understand the data structure first
    #for i in range(N):
    #   for j in range(T):
    #        temp[j,:] = W[x[i,j],:]
    #    out[i,:,:] = temp    
        
    ##@@@ very fancy techniques by Github from fancy indexing
    ## b = [[1.5,2,3],[4,5,6]]
    ## b[[1, 0, 1, 0]][:,[0,1,2,0]] 
    ## : make this indexing different. Choose every row from the matrix
    # For each element in x we get its corresponding word embedding vector from W.
    out = W[x, :]
    cache = x, W
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    #pass
    N,T,D = dout.shape
    x,W = cache
    ###@@@ do not know how to do derivative when x is the index of w, makes it out
    dW = np.zeros_like(W)
    # Adds the upcoming gradients into the corresponding index from W.
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # pass output size A(N,4H)
    A = np.dot(x,Wx)+np.dot(prev_h,Wh) + b
    a1,a2,a3,a4 = np.split(A,4,axis=1)
    i = sigmoid(a1)
    f = sigmoid(a2)
    o = sigmoid(a3)
    g = np.tanh(a4)
    next_c = f*prev_c + i*g
    next_h = o * np.tanh(next_c) 
    
    cache = x,prev_h,prev_c,Wx,Wh,b,next_c,next_h,i,f,o,g
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # pass
    ### @@@ the matrix were split into four groups
    ### @@ solution: np.concatenate
    # unpack cache
    x,prev_h,prev_c,Wx,Wh,b,next_c,next_h,i,f,o,g = cache
    
    # initialization 
    dx = np.zeros_like(x)
    dprev_h = np.zeros_like(prev_h)
    dprev_c = np.zeros_like(prev_c)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    
    # backprogate through the multiply gate
    dh_do = np.tanh(next_c)*dnext_h
    
    
    do_da3 = o*(1-o)
    dh_da3 = dh_do*do_da3
    
    dh_dnext_c = o*((1 - np.square(np.tanh(next_c))))*dnext_h
    
    # combine two gradient flow
    ### @@@ based on the computational graph: two flows combines!
    dnext_combine =  dnext_c + dh_dnext_c
    
    # Backprop dtanh_dc to calculate dprev_c.
    dprev_c = f*(dnext_combine)
    
    ##@@@ when doing back progation, it is safe to multipy the upstream gradients!! easy to forget( * dnext_combine)
    ###@@@ when back progate, according to the computational graph which really helps 
    
    # Backprop dtanh_dc towards each gate.
    dnext_combine_df = prev_c * dnext_combine 
    dnext_combine_di = g * dnext_combine
    dnext_combine_dg = i * dnext_combine
    
    
    # Backprop through gate activation functions to calculate gate derivatives.
    df_da2 = f*(1-f)
    dnext_combine_da2 = dnext_combine_df * df_da2
    di_da1 = i*(1-i)
    dnext_combine_da1 = dnext_combine_di*di_da1
    dg_da4 = 1 - g**2
    dnext_combine_da4 = dnext_combine_dg * dg_da4
    
    # Concatenate back up our 4 gate derivatives to get da_vector.
    dcombine_dA = np.concatenate((dnext_combine_da1,dnext_combine_da2,dh_da3,dnext_combine_da4),axis=1)
    
    # -------------------
    ###@@@ db is a colomn vector
    db = np.sum(dcombine_dA,axis=0) # (4H)
    dx = np.dot(dcombine_dA,Wx.T) #(N, D)  (N,4H) (D, 4H)
    dWx = np.dot(x.T,dcombine_dA)
    dWh = np.dot(prev_h.T,dcombine_dA)
    dprev_h = np.dot(dcombine_dA,Wh.T)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # pass
    N,T,D  = x.shape
    N,H = h0.shape
    ### take-away pay attention to the interface: the dimention of the matrix
    ### intialization
    prev_h = h0
    prev_c = np.zeros_like(prev_h)
    
    h = np.zeros([N,T,H])    
    cache = [] 
    ###@@@ how about prev_c
    ###@@@ just intialize it with all zeros
    for i in range(T):
        cur_x = x[:,i,:]    
        next_h,next_c,cache_temp = lstm_step_forward(cur_x, prev_h, prev_c, Wx, Wh, b)
        h[:,i,:] = next_h
        prev_h = next_h
        prev_c = next_c
        cache.append(cache_temp)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # pass
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    
    
    #1) backward from the T to T-1 , T-2
    #2) shared weights, the gradients should be added / accumulated
    #3) @@@ [Add] the current timestep upstream gradient to previous calculated dh.
    # based on the computional graph
    #intialization
    x,prev_h,prev_c,Wx,Wh,b,next_c,next_h,i,f,o,g = cache[0]
    
    N,T,H = dh.shape
    N,D = x.shape
    
    dx = np.zeros([N, T, D])
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(prev_h)
    
    # Initial gradient for cell is all zero.
    dnext_c = np.zeros_like(prev_c)
    
    for i in reversed(range(T)):
        cur_h = dh[:,i,:] + dprev_h
        cur_cache = cache[i]
        dx_temp, dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(cur_h, dnext_c, cur_cache)
        # for recursion
        #cur_h = dprev_h
        dnext_c = dprev_c
        # for gradients
        dx[:,i,:]=dx_temp
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    if(len(x.shape) is 3):
        N, T, D = x.shape
    else:
        D = x.shape
        N =1
        T =1 
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
