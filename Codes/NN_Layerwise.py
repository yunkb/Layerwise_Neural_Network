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
    def __init__(self, hyper_p, run_options, data_dimension, labels_dimension, weight_layer_counter):
        
###############################################################################
#                    Constuct Neural Network Architecture                     #
###############################################################################        
        # Initialize placeholders
        self.data_tf = tf.placeholder(tf.float32, shape=[None, data_dimension], name = "data_tf")
        self.labels_tf = tf.placeholder(tf.float32, shape=[None, labels_dimension], name = "labels_tf") # This is needed for batching during training, else can just use state_data
                           
        # Initialize weights and biases storage
        self.layers = [data_dimension] + [hyper_p.num_hidden_nodes]*weight_layer_counter + [labels_dimension]
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)  
        
        ############################
        #   Initial Architecture   #
        ############################
        if weight_layer_counter == 0: # If first iteration, initialize output layer
            for l in range(0, 2):
                W = tf.get_variable("W" + str(l+1), dtype = tf.float32, shape = [self.layers[l], self.layers[l + 1]], initializer = tf.random_normal_initializer())
                b = tf.get_variable("b" + str(l+1), dtype = tf.float32, shape = [1, self.layers[l + 1]], initializer = tf.constant_initializer(0))                                  
                tf.summary.histogram("weights" + str(l+1), W)
                tf.summary.histogram("biases" + str(l+1), b)
                self.weights.append(W)
                self.biases.append(b)
        
        ##############################
        #   Extending Architecture   #
        ##############################   
        if weight_layer_counter > 0: # Load pre-trained hidden layer weights and output layer weights
            graph = tf.get_default_graph()
            for l in range(0, weight_layer_counter):
                W = graph.get_tensor_by_name("NN_layerwise/W" + str(l+1) + ':0')
                b = graph.get_tensor_by_name("NN_layerwise/b" + str(l+1) + ':0')
                self.weights.append(W)
                self.biases.append(b)
            self.weights.append(W) # add a copy of the output weights which also extends list of weights
        
        # Initialize new hidden layer as 0           
        with tf.variable_scope("NN_layerwise") as scope: 
            l = weight_layer_counter
            W = tf.get_variable("W" + str(l+1), dtype = tf.float32, shape = [self.layers[l], self.layers[l + 1]], initializer = tf.constant_initializer(0))
            b = tf.get_variable("b" + str(l+1), dtype = tf.float32, shape = [1, self.layers[l + 1]], initializer = tf.constant_initializer(0))                                  
            tf.summary.histogram("weights" + str(l+1), W)
            tf.summary.histogram("biases" + str(l+1), b)
            self.weights[l] = W
            self.biases[l] = b
        
        # Dictionary of weights and biases to train
        
        # Ensures train.Saver only saves the weights and biases                
        self.saver_NN_layerwise = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "NN_layerwise")
                
###############################################################################
#                           Network Propagation                               #
###############################################################################                  
        self.prediction = self.forward_prop(self.data_tf, num_layers)                
        self.classify = tf.argmax(self.prediction, 1)
        
###############################################################################
#                                Methods                                      #
############################################################################### 
    def forward_prop(self, X, num_layers):  
        with tf.variable_scope("fwd_prop") as scope:
            for l in range(0, num_layers-2):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.nn.relu(tf.add(tf.matmul(X, W), b))
                #tf.summary.histogram("activation" + str(l+1), X)
            W = self.weights[-1]
            b = self.biases[-1]
            output = tf.add(tf.matmul(X, W), b)
            return output
    
