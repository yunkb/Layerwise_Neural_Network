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

    biases_init_value = 0  
        
    for l in range(0, len(NN.layers)-1): 
        z_W = tf.get_variable("z_W" + str(l+1), dtype = tf.float32, shape = [NN.layers[l], NN.layers[l + 1]], initializer = tf.random_normal_initializer(), trainable=False)
        z_b = tf.get_variable("z_b" + str(l+1), dtype = tf.float32, shape = [1, NN.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value), trainable=False)                                  
        z_weights.append(z_W)
        z_biases.append(z_b)
        
        lagrange_W = tf.get_variable("lagrange_W" + str(l+1), dtype = tf.float32, shape = [NN.layers[l], NN.layers[l + 1]], initializer = tf.random_normal_initializer(), trainable=False)
        lagrange_b = tf.get_variable("lagrange_b" + str(l+1), dtype = tf.float32, shape = [1, NN.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value), trainable=False)                                  
        lagrange_weights.append(lagrange_W)
        lagrange_biases.append(lagrange_b)
        
    return z_weights, z_biases, lagrange_weights, lagrange_biases

###############################################################################
#                               ADMM Penalty Term                             #
###############################################################################
def ADMM_penalty_term(NN, pen, z_weights, z_biases, lagrange_weights, lagrange_biases):
    ADMM_penalty = 0.0
    for l in range(0, len(NN.weights)):  
        weights_norm = pen/2 * tf.pow(tf.norm(NN.weights[l] - z_weights[l] + lagrange_weights[l]/pen, 2), 2)
        biases_norm = pen/2 * tf.pow(tf.norm(NN.biases[l] - z_biases[l] + lagrange_biases[l]/pen, 2), 2)
        ADMM_penalty += weights_norm + biases_norm
        
    return ADMM_penalty

###############################################################################
#                      Update z and Lagrange Multiplier                       #
###############################################################################
def update_z_and_lagrange_multiplier(sess, NN, alpha, pen, z_weights, z_biases, lagrange_weights, lagrange_biases):   
    for l in range(0, len(NN.weights)):  
        sess.run(tf.assign(z_weights[l], soft_threshold_weights(NN, lagrange_weights, lagrange_biases, alpha, pen)))
        sess.run(tf.assign(z_biases[l], soft_threshold_biases(NN, lagrange_weights, lagrange_biases, alpha, pen)))
        sess.run(tf.assign(lagrange_weights[l], pen*(NN.weights[l] - z_weights[l])))
        sess.run(tf.assign(lagrange_biases[l], pen*(NN.biases[l] - z_biases[l])))
      
###############################################################################
#                       Soft Thresholding Operator                            #
###############################################################################   
def soft_threshold_weights(NN, lagrange_weights, lagrange_biases, alpha, pen):
    weights_val = []
    for l in range(0, len(NN.weights)):
        val = NN.weights[l] + lagrange_weights[l]/pen
        weights_val.append(val)
        
        # annoying digital logic workaround to implement conditional.
        # construct vectors of 1's and 0's that we can multiply
        # by the proper value and sum together
        cond1 = tf.where(tf.greater(weights_val, alpha/pen), tf.ones((NN.layers[l], NN.layers[l + 1])), tf.zeros((NN.layers[l], NN.layers[l + 1])))
        cond3 = tf.where(tf.less(weights_val, -1.0*alpha/pen), tf.ones((NN.layers[l], NN.layers[l + 1])), tf.zeros((NN.layers[l], NN.layers[l + 1])))
        # cond2 is not needed since the complement of the intersection
        # of (cond1 and cond3) is cond2 and already assigned to 0
        
        z_weights = cond1*(weights_val[l] - alpha/pen) + cond3*(weights_val[l] + alpha/pen)
    
    return z_weights

def soft_threshold_biases(NN, lagrange_weights, lagrange_biases, alpha, pen):
    biases_val = []
    for l in range(0, len(NN.weights)):
        val = NN.biases[l] + lagrange_biases[l]/pen
        biases_val.append(val)
        
        # annoying digital logic workaround to implement conditional.
        # construct vectors of 1's and 0's that we can multiply
        # by the proper value and sum together
        cond1 = tf.where(tf.greater(biases_val, alpha/pen), tf.ones((1, NN.layers[l + 1])), tf.zeros((1, NN.layers[l + 1])))
        cond3 = tf.where(tf.less(biases_val, -1.0*alpha/pen), tf.ones((1, NN.layers[l + 1])), tf.zeros((1, NN.layers[l + 1])))
        # cond2 is not needed since the complement of the intersection
        # of (cond1 and cond3) is cond2 and already assigned to 0
        
        z_biases = cond1*(biases_val[l] - alpha/pen) + cond3*(biases_val[l] + alpha/pen)
    
    return z_biases

    



