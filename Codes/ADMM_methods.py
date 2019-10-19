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

    weights_init_value = 0.05
    biases_init_value = 0  
        
    for l in range(0, len(NN.layers)-1): 
        z_W = tf.get_variable("z_W" + str(l+1), dtype = tf.float64, shape = [NN.layers[l], NN.layers[l + 1]], initializer = tf.random_normal_initializer(), trainable=False)
        z_b = tf.get_variable("z_b" + str(l+1), dtype = tf.float64, shape = [1, NN.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value), trainable=False)                                  
        z_weights.append(z_W)
        z_biases.append(z_b)
        
        lagrange_W = tf.get_variable("lagrange_W" + str(l+1), dtype = tf.float64, shape = [NN.layers[l], NN.layers[l + 1]], initializer = tf.random_normal_initializer(), trainable=False)
        lagrange_b = tf.get_variable("lagrange_b" + str(l+1), dtype = tf.float64, shape = [1, NN.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value), trainable=False)                                  
        lagrange_weights.append(lagrange_W)
        lagrange_biases.append(lagrange_b)
        
    return z_weights, z_biases, lagrange_weights, lagrange_biases

###############################################################################
#                      Update z and Lagrange Multiplier                       #
###############################################################################
def update_z_and_lagrange_multiplier(sess, NN, num_layers, z_weights, z_biases, lagrange_weights, lagrange_biases, pen):   
    for l in range(0, len(NN.weights)):  
        sess.run(tf.assign(z_weights[l], pen*(NN.weights)))
        sess.run(tf.assign(z_biases[l], biases_current[l] + perturb_h*rand_v_biases[l]))
        sess.run(tf.assign(lagrange_weights[l], pen*(NN.weights[l] - z_weights[l])))
        sess.run(tf.assign(lagrange_biases[l], pen*(NN.biases[l] - z_biases[l])))
      
###############################################################################
#                                 Compute z                                   #
###############################################################################   
def compute_z(NN, lagrange_weights, lagrange_biases, alpha, pen):
    val = []
    for l in range(0, len(NN.weights)):
        weights_val = NN.weights[l] + lagrange_weights[l] / pen
        biases_val = NN.biases[l] + lagrange_biases[l] / pen
        
    # annoying digital logic workaround to implement conditional.
    # construct vectors of 1's and 0's that we can multiply
    # by the proper value and sum together
    cond1 = tf.where(tf.greater(val, alpha/pen), tf.ones((hyper_p.N_r, 1)), tf.zeros((hyper_p.N_r, 1)))
    cond3 = tf.where(tf.less(val, -1.0*alpha/pen), tf.ones((hyper_p.N_r, 1)), tf.zeros((hyper_p.N_r, 1)))
    # cond2 is not needed since the complement of the intersection
    # of (cond1 and cond3) is cond2 and already assigned to 0
    
    dummy_z = cond1*(val - alpha/pen) + cond3*(val + alpha/pen)
    
    return dummy_z
    



