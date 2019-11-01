#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:00:22 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                           Constuct ADMM Objects                             #
###############################################################################        
def construct_ADMM_objects(NN):
    z_weights = [] # This will be a list of tensorflow variables
    z_biases = [] # This will be a list of tensorflow variables
    
    lagrange_weights = [] # This will be a list of tensorflow variables
    lagrange_biases = [] # This will be a list of tensorflow variables
    
    #=== Convolutional Hidden Layers ===#    
    for l in range(1, len(NN.layers)-1): 
        z_W = tf.get_variable("z_W" + str(l), dtype = tf.float32, shape = [NN.layers[l][0], NN.layers[l][0], NN.layers[l-1][1], NN.layers[l][1]], initializer = tf.random_normal_initializer(), trainable=False)
        z_b = tf.get_variable("z_b" + str(l), dtype = tf.float32, shape = [NN.layers[l][1]], initializer = tf.constant_initializer(0), trainable=False)                                  
        z_weights.append(z_W)
        z_biases.append(z_b)
        
        lagrange_W = tf.get_variable("lagrange_W" + str(l), dtype = tf.float32, shape = [NN.layers[l][0], NN.layers[l][0], NN.layers[l-1][1], NN.layers[l][1]], initializer = tf.constant_initializer(0), trainable=False)
        lagrange_b = tf.get_variable("lagrange_b" + str(l), dtype = tf.float32, shape = [NN.layers[l][1]], initializer = tf.constant_initializer(0), trainable=False)                                  
        lagrange_weights.append(lagrange_W)
        lagrange_biases.append(lagrange_b)
    
    #=== Fully Connected Output Layer ===#
    l += 1
    z_W = tf.get_variable("z_W" + str(l), dtype = tf.float32, shape = [NN.num_features, NN.layers[l]], initializer = tf.random_normal_initializer(), trainable=False)
    z_b = tf.get_variable("z_b" + str(l), dtype = tf.float32, shape = [1,NN.layers[l]], initializer = tf.constant_initializer(0), trainable=False)                                  
    z_weights.append(z_W)
    z_biases.append(z_b)
    
    lagrange_W = tf.get_variable("lagrange_W" + str(l), dtype = tf.float32, shape = [NN.num_features, NN.layers[l]], initializer = tf.constant_initializer(0), trainable=False)
    lagrange_b = tf.get_variable("lagrange_b" + str(l), dtype = tf.float32, shape = [1,NN.layers[l]], initializer = tf.constant_initializer(0), trainable=False)                                  
    lagrange_weights.append(lagrange_W)
    lagrange_biases.append(lagrange_b)
        
    return z_weights, z_biases, lagrange_weights, lagrange_biases

###############################################################################
#                      Update z and Lagrange Multiplier                       #
###############################################################################
def update_z_and_lagrange_multiplier_tf_operations(NN, alpha, pen, z_weights, z_biases, lagrange_weights, lagrange_biases):   
    for l in range(0, len(NN.weights)):  
        tf.assign(z_weights[l], NN.weights[l], name = "z_weights_initial_value" + str(l+1))
        tf.assign(z_biases[l], NN.biases[l], name = "z_biases_initial_value" + str(l+1))
        
        tf.assign(z_weights[l], soft_threshold_weights(NN, l, lagrange_weights, lagrange_biases, alpha, pen), name = "z_weights_update" + str(l+1))
        tf.assign(z_biases[l], soft_threshold_biases(NN, l, lagrange_weights, lagrange_biases, alpha, pen), name = "z_biases_update" + str(l+1))
        tf.assign(lagrange_weights[l], lagrange_weights[l] + pen*(NN.weights[l] - z_weights[l]), name = "lagrange_weights_update" + str(l+1))
        tf.assign(lagrange_biases[l], lagrange_biases[l] + pen*(NN.biases[l] - z_biases[l]), name = "lagrange_biases_update" + str(l+1))
      
###############################################################################
#                       Soft Thresholding Operator                            #
###############################################################################   
def soft_threshold_weights(NN, l, lagrange_weights, lagrange_biases, alpha, pen):
    if l < len(NN.weights)-1: # Convolutional hidden layers
        weight_shape = [NN.layers[l+1][0], NN.layers[l+1][0], NN.layers[l][1], NN.layers[l+1][1]]
    if l == len(NN.weights)-1: # Fully connected output layer
        weight_shape = [NN.num_features, NN.layers[l + 1]]
    
    weights_val = NN.weights[l] + lagrange_weights[l]/pen
    
    cond1 = tf.where(tf.greater(weights_val, alpha/pen), tf.ones(weight_shape), tf.zeros(weight_shape))
    cond3 = tf.where(tf.less(weights_val, -1.0*alpha/pen), tf.ones(weight_shape), tf.zeros(weight_shape))
    
    new_z_weights = cond1*(weights_val - alpha/pen) + cond3*(weights_val + alpha/pen)
    
    return new_z_weights

def soft_threshold_biases(NN, l, lagrange_weights, lagrange_biases, alpha, pen):
    if l < len(NN.weights)-1: # Convolutional hidden layers
        weight_shape = [NN.layers[l+1][1]]
    if l == len(NN.weights)-1: # Fully connected output layer
        weight_shape = [1, NN.layers[l + 1]]
        
    biases_val = NN.biases[l] + lagrange_biases[l]/pen
    
    cond1 = tf.where(tf.greater(biases_val, alpha/pen), tf.ones(weight_shape), tf.zeros(weight_shape))
    cond3 = tf.where(tf.less(biases_val, -1.0*alpha/pen), tf.ones(weight_shape), tf.zeros(weight_shape))

    new_z_biases = cond1*(biases_val - alpha/pen) + cond3*(biases_val + alpha/pen)
    
    return new_z_biases

    



