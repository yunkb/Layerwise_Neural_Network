#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:27:35 2019

@author: hwan
"""

import numpy as np
import tensorflow as tf
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                    Initialize z and Lagrange Multiplier                     # 
###############################################################################
def initialize_z_and_lagrange_multiplier(weights):
    z = weights
    lagrange = []
    for l in range(0, len(weights)):
        lagrange.append(np.zeros(weights[l].shape))
    
    return z, lagrange

###############################################################################
#                       Update ADMM penaltyalty Term                          # 
###############################################################################
def update_ADMM_penalty_terms(penalty, weights_tf, z, lagrange):
    ADMM_penalty = 0.0
    for l in range(0, len(weights_tf)):  
        ADMM_penalty += penalty/2 * tf.pow(tf.norm(weights_tf[l] - z[l] + lagrange[l]/penalty, 2), 2)

    return ADMM_penalty
        
###############################################################################
#                      Update z and Lagrange Multiplier                       #
###############################################################################
def update_z_and_lagrange_multiplier(weights, alpha, penalty, z, lagrange):   
    for l in range(0, len(weights)):  
        z[l] = soft_threshold_weights(weights, l, lagrange, alpha, penalty)
        lagrange[l] += penalty*(weights[l] - z[l])
        
    return z, lagrange
      
###############################################################################
#                       Soft Thresholding Operator                            #
###############################################################################   
def soft_threshold_weights(weights, l, lagrange, alpha, penalty):
    weights_val = weights[l] + lagrange[l]/penalty
    
    cond1 = tf.where(tf.greater(weights_val, alpha/penalty), tf.ones(weights[l].shape), tf.zeros(weights[l].shape))
    cond3 = tf.where(tf.less(weights_val, -1.0*alpha/penalty), tf.ones(weights[l].shape), tf.zeros(weights[l].shape))
    
    new_z_weights = cond1*(weights_val - alpha/penalty) + cond3*(weights_val + alpha/penalty)
    
    return new_z_weights

