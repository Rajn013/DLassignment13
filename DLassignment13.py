#!/usr/bin/env python
# coding: utf-8

# Why is it generally preferable to use a Logistic Regression classifier rather than a classical Perceptron (i.e., a single layer of linear threshold units trained using the Perceptron training algorithm)? How can you tweak a Perceptron to make it equivalent to a Logistic Regression classifier?
# 

# Logistic Regression provides probabilistic outputs, estimating the probability of an instance belonging to a class, while a Perceptron gives binary outputs.
# Logistic Regression produces continuous output, allowing for more nuanced decision-making, whereas a Perceptron has discrete outputs based on a threshold.
# Logistic Regression uses differentiable training with a sigmoid activation function, enabling efficient gradient-based optimization, whereas a Perceptron uses discrete updates.
# Logistic Regression can model complex decision boundaries, whereas a Perceptron is limited to linear decision boundaries.
# 
# To make a Perceptron equivalent to a Logistic Regression classifier:
# 
# Replace the step function with the sigmoid (logistic) function as the activation function.
# Use differentiable training techniques like gradient descent for optimization.
# Incorporate techniques like feature engineering or kernel methods to handle nonlinear decision boundaries.
# 

# Why was the logistic activation function a key ingredient in training the first MLPs?
# 

# The logistic activation function was differentiable, allowing for efficient gradient-based optimization using techniques like backpropagation.
# It produced outputs between 0 and 1, making it suitable for modeling probabilities or binary classification problems.
# The logistic function introduced nonlinearity, enabling MLPs to learn and represent complex relationships in the data.
# Its smooth and continuous transition between 0 and 1 facilitated smoother optimization and avoided issues like vanishing or exploding gradients.
# The logistic activation function was widely used in the early days of neural networks, playing a significant role in their development and success.

# Name three popular activation functions. Can you draw them?
# 

# Sigmoid (Logistic) Activation Function:
# Range: The sigmoid function maps the input to a value between 0 and 1.
# Shape: It has an S-shaped curve with a smooth and continuous transition.
# Formula: σ(x) = 1 / (1 + exp(-x))
# Plot:
#     
#     
#     
#     
#        1 |                  
#          |         ___      
#          |       /       
#        0.5 |     /        
#          |    /          
#          |  _/           
#          | /             
#          |/              
#        0 |________________
#          -5  -4  -3  -2  -1   0   1   2   3   4   5
# 
#      
#     
# Rectified Linear Unit (ReLU) Activation Function:
# Range: The ReLU function outputs the input as it is if it's positive, otherwise outputs 0.
# Shape: It has a linear increase for positive values and remains constant at 0 for negative values.
# Formula: f(x) = max(0, x)
# Plot:
#     
#     
#     
#           1 |                
#          |                
#          |                
#        0.5 |                
#          |                
#          | __             
#          |/               
#        0 |________________
#          -5  -4  -3  -2  -1   0   1   2   3   4   5
#  
#     
#     
# Hyperbolic Tangent (tanh) Activation Function:
# Range: The tanh function maps the input to a value between -1 and 1.
# Shape: It has an S-shaped curve similar to the sigmoid function but centered around 0.
# Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# Plot:
# 
#     
#     
#     
#            1 |                  
#          |         ___      
#          |       /       
#        0.5 |     /        
#          |    /          
#          |  _/           
#          |/              
#          | __             
#       -0.5 |/               
#          |                
#          |                
#        0 |________________
#          -5  -4  -3  -2  -1   0   1   2   3   4   5
# 

# Suppose you have an MLP composed of one input layer with 10 passthrough neurons, followed by one hidden layer with 50 artificial neurons, and finally one output layer with 3 artificial neurons. All artificial neurons use the ReLU activation function.
# What is the shape of the input matrix X?
# 

# the shape of the input matrix X would be (N, 10) for the MLP with one input layer of 10 passthrough neurons.

# What about the shape of the hidden layer’s weight vector Wh, and the shape of its bias vector bh?
# 

# The shape of the weight vector Wh connecting the input layer to the hidden layer would be (10, 50). It has 10 rows (corresponding to the input neurons) and 50 columns (corresponding to the hidden neurons).
# 
# The shape of the bias vector bh for the hidden layer would be (50,). It is a 1-dimensional vector with 50 elements, one for each neuron in the hidden layer.

# What is the shape of the output layer’s weight vector Wo, and its bias vector bo?

# the shape of the weight vector Wo connecting the hidden layer to the output layer would be (50, 3), and the shape of the bias vector bo for the output layer would be (3,).

# What is the shape of the network’s output matrix Y?

# the shape of the network's output matrix Y would be (N, 3) for the MLP with one output layer of 3 neurons.

# Write the equation that computes the network’s output matrix Y as a function of X, Wh, bh, Wo and bo.
#         

# The equation calculates the output matrix Y by multiplying the input matrix X with the weight matrix Wh, applying the ReLU activation function, multiplying the result with the weight matrix Wo, and adding the bias vectors bh and bo.

# How many neurons do you need in the output layer if you want to classify email into spam or ham? What activation function should you use in the output layer? If instead you want to tackle MNIST, how many neurons do you need in the output layer, using what activation function?
# 

# For classifying emails into spam or ham, you would need 2 neurons in the output layer, and the sigmoid or softmax activation function can be used.
# For tackling the MNIST dataset, you would need 10 neurons in the output layer, and the softmax activation function is commonly used.

# What is backpropagation and how does it work? What is the difference between backpropagation and reverse-mode autodiff?

# Backpropagation is an algorithm for training neural networks by efficiently computing the gradients of the network parameters with respect to the loss function. It works by performing a forward pass to compute the network's predictions, followed by a backward pass to calculate the gradients. The gradients are then used to update the network parameters and improve its performance.
# 
# Reverse-mode autodiff is a more general technique for computing gradients in computational graphs. It encompasses backpropagation as a specific case. While backpropagation is tailored for neural networks, reverse-mode autodiff can be applied to various computational graphs. Backpropagation is a specific implementation of reverse-mode autodiff for neural networks.

# Can you list all the hyperparameters you can tweak in an MLP? If the MLP overfits the training data, how could you tweak these hyperparameters to try to solve the problem?
# 

# Number of hidden layers
# Number of neurons per hidden layer
# Activation function
# Learning rate
# Regularization
# Dropout
# Batch size
# Number of training epochs
# Weight initialization
# Optimization algorithm
# Early stopping
# 
# 
# Decrease the model's capacity (neurons or layers).
# Increase regularization strength.
# Apply dropout more aggressively.
# Decrease the learning rate.
# Increase the batch size.
# Implement early stopping.
# Experimenting with these hyperparameters can help mitigate overfitting and improve the generalization of the MLP.

# Train a deep MLP on the MNIST dataset and see if you can get over 98% precision. Try adding all the bells and whistles (i.e., save checkpoints, restore the last checkpoint in case of an interruption, add summaries, plot learning curves using TensorBoard, and so on).

# TensorFlow installed (pip install tensorflow).
# MNIST dataset downloaded or imported using TensorFlow (from tensorflow.keras.datasets import mnist).
# 
# 
# The MNIST dataset is loaded and preprocessed. The images are reshaped to 784-dimensional vectors and normalized to values between 0 and 1.
# 
# The MLP architecture is defined using the Sequential API from TensorFlow. It consists of three fully connected (Dense) layers with ReLU activation functions. Dropout layers are added to prevent overfitting.
# 
# The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
# 
# Checkpoints are set up using the ModelCheckpoint callback. The best model weights are saved to a file (model_checkpoint.h5) based on the validation accuracy.
# 
# TensorBoard is configured using the TensorBoard callback, specifying the directory to save the log files (./logs).
# 
# The model is trained using the fit function, with the training and validation data, batch size, number of epochs, and the defined callbacks.
# 
# After training, the model is evaluated on the test data using the evaluate function. The test loss and accuracy are printed.
