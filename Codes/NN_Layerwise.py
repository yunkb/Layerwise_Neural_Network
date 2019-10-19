#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class Layerwise:
    def __init__(self, hyper_p, run_options, data_dimension, labels_dimension, construct_flag):
        
###############################################################################
#                    Constuct Neural Network Architecture                     #
###############################################################################        
        # Initialize placeholders
        self.data_train_tf = tf.placeholder(tf.float32, shape=[None, data_dimension], name = "data_train_tf")
        self.labels_train_tf = tf.placeholder(tf.float32, shape=[None, labels_dimension], name = "labels_train_tf") # This is needed for batching during training, else can just use state_data
            
        self.data_test_tf = tf.placeholder(tf.float32, shape=[None, data_dimension], name = "data_test_tf")
        self.labels_test_tf = tf.placeholder(tf.float32, shape=[None, labels_dimension], name = "labels_test_tf") # This is needed for batching during training, else can just use state_data
       
        # Initialize weights and biases
        self.layers = [data_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [labels_dimension]
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)
        biases_init_value = 0       
        
        # Construct weights and biases
        if construct_flag == 1:
            with tf.variable_scope("NN_layerwise") as scope:
                for l in range(0, num_layers-1): 
                    W = tf.get_variable("W" + str(l+1), dtype = tf.float32, shape = [self.layers[l], self.layers[l + 1]], initializer = tf.random_normal_initializer())
                    b = tf.get_variable("b" + str(l+1), dtype = tf.float32, shape = [1, self.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value))                                  
                    tf.summary.histogram("weights" + str(l+1), W)
                    tf.summary.histogram("biases" + str(l+1), b)
                    self.weights.append(W)
                    self.biases.append(b)
            
            # Ensures train.Saver only saves the weights and biases                
            self.saver_NN_layerwise = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "NN_layerwise")
        
        # Load trained model  
        if construct_flag == 0: 
            graph = tf.get_default_graph()
            for l in range(0, hyper_p.truncation_layer):
                W = graph.get_tensor_by_name("NN_layerwise/W" + str(l+1) + ':0')
                b = graph.get_tensor_by_name("NN_layerwise/b" + str(l+1) + ':0')
                self.weights.append(W)
                self.biases.append(b)
                
###############################################################################
#                           Network Propagation                               #
###############################################################################                  
        # Training and Testing Placeholders
        self.prediction_train = self.forward_prop(self.data_train_tf, num_layers)        
        self.prediction_test = self.forward_prop(self.data_test_tf, num_layers)
        
###############################################################################
#                                Methods                                      #
############################################################################### 
    def forward_prop(self, X, num_layers):  
        with tf.variable_scope("fwd_prop") as scope:
            for l in range(0, num_layers-2):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.tanh(tf.add(tf.matmul(X, W), b))
                #tf.summary.histogram("activation" + str(l+1), X)
            W = self.weights[-1]
            b = self.biases[-1]
            output = tf.add(tf.matmul(X, W), b)
            return output
    
