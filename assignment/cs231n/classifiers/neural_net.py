import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  layer1_scores = np.dot(X,W1) + b1
  layer1_scores = np.maximum(layer1_scores, np.zeros(layer1_scores.shape))
  scores = np.dot(layer1_scores, W2) + b2
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
   # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  num_dim = X.shape[1]
  num_classes = scores.shape[1]
  num_hidden = W1.shape[1]

  exp_scores = np.exp(scores)
  #print exp_scores
  correct_label_exp_scores = exp_scores[np.array(range(num_train)),y]
  #print correct_label_exp_scores
  exp_scores_sum = np.sum(exp_scores, axis=1)
  #print exp_scores_sum
  loss = -1 * np.sum(np.log(correct_label_exp_scores / exp_scores_sum)) / num_train
  #print loss
  loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2*W2))
  #print loss

  #dw2
  # compute the gradients
  grads = {}
  grad_weight = exp_scores / exp_scores_sum.reshape(num_train,1)
  grad_weight[np.array(range(num_train)),y] -= np.ones(num_train)
  grad_weight = grad_weight.reshape(num_train, 1, num_classes)
  dW2_coef = layer1_scores.reshape(num_train, num_hidden, 1)
  dW2 = np.sum(dW2_coef * grad_weight, axis = 0) / num_train
  dW2 += reg * W2
  grads['W2'] = dW2
  db2_coef = np.ones(num_train).reshape(num_train,1,1)
  db2 = np.sum(db2_coef * grad_weight, axis=0)/num_train
  grads['b2'] = db2


  #(exp_score (N*C) 1st row/ row_sum -correct_lable) * W2.T).sum(axis=0
  #grad_back computes the back propogation based derivatives. 
  # RECLU(X * W1 + b1) is grad_back
  grad_back = exp_scores / exp_scores_sum.reshape(num_train, 1)
  #print grad_back.shape
  grad_back[np.array(range(num_train)), y] -= np.ones(num_train)  
  grad_back = np.dot(grad_back, W2.T)
  hidden_scores = np.dot(X, W1) + b1
  grad_back_status = np.maximum(hidden_scores / np.abs(hidden_scores), np.zeros(hidden_scores.shape)) 
  grad_back *= grad_back_status
  grad_back = grad_back.reshape(num_train, num_hidden, 1)
  dW1_coeff = X.reshape(num_train, 1, num_dim)
  dW1 = grad_back * dW1_coeff
  dW1 = np.sum(dW1, axis=0) / num_train
  dW1 = dW1.T + reg * W1
  grads['W1'] = dW1
  grads['b1'] = np.sum(grad_back, axis=0).reshape(num_hidden) / num_train

  return loss, grads

