#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class FullyConnectedLayerwise:
    def __init__(self, hyper_p, hidden_layer_counter, data_dimension, labels_dimension, savefilepath):
        
###############################################################################
#                    Constuct Neural Network Architecture                     #
###############################################################################        
        #=== Construct Placeholders ===#
        self.data_tf = tf.placeholder(tf.float32, shape=[None, data_dimension], name = "data_tf")
        self.labels_tf = tf.placeholder(tf.float32, shape=[None, labels_dimension], name = "labels_tf") # This is needed for batching during training, else can just use state_data
                           
        #=== Define Architecture and Create Parameter Storage ===#
        self.layers = [data_dimension] + [data_dimension]*(hidden_layer_counter) + [labels_dimension]
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)  
        
        ############################
        #   Initial Architecture   #
        ############################
        # If first iteration, initialize output layer
        if hidden_layer_counter == 1: 
            with tf.variable_scope("NN") as scope: 
                for l in range(1, 3):
                    W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.random_normal_initializer())
                    b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(0))                                  
                    tf.summary.histogram("weights" + str(l), W)
                    tf.summary.histogram("biases" + str(l), b)
                    self.weights.append(W)
                    self.biases.append(b)
        
        ##############################
        #   Extending Architecture   #
        ##############################   
        if hidden_layer_counter > 1: 
            with tf.variable_scope("NN") as scope: 
                # Load pre-trained weights and biases
                for l in range(1, hidden_layer_counter):
                    df_trained_weights = pd.read_csv(savefilepath + "_W" + str(l) + '.csv')
                    df_trained_biases = pd.read_csv(savefilepath + "_b" + str(l) + '.csv')
                    restored_W = df_trained_weights.values.reshape([self.layers[l-1], self.layers[l]])
                    restored_b = df_trained_biases.values.reshape([1, self.layers[l]])
                    W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.constant_initializer(restored_W), trainable = False)
                    b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(restored_b), trainable = False)                                  
                    self.weights.append(W)
                    self.biases.append(b)
                
                # Initialize new hidden layer weights and biases as 0           
                l = hidden_layer_counter
                W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.constant_initializer(0))
                b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(0))                                  
                tf.summary.histogram("weights" + str(l), W)
                tf.summary.histogram("biases" + str(l), b)
                self.weights.append(W)
                self.biases.append(b)
                    
                # Load pre-trained output layer weights and biases. Note these wlll be trained again
                l = hidden_layer_counter + 1
                df_trained_weights = pd.read_csv(savefilepath + "_Woutput" + '.csv')
                df_trained_biases = pd.read_csv(savefilepath + "_boutput" + '.csv')
                restored_W = df_trained_weights.values.reshape([self.layers[l-1], self.layers[l]])
                restored_b = df_trained_biases.values.reshape([1, self.layers[l]])
                W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.constant_initializer(restored_W))
                b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(restored_b))                                  
                tf.summary.histogram("weights" + str(l), W)
                tf.summary.histogram("biases" + str(l), b)
                self.weights.append(W)
                self.biases.append(b)
        
###############################################################################
#                           Network Propagation                               #
###############################################################################                  
        self.logits = self.forward_prop(self.data_tf, num_layers)                
        
###############################################################################
#                                Methods                                      #
############################################################################### 
    def forward_prop(self, X, num_layers):  
        with tf.variable_scope("fwd_prop") as scope:
            for l in range(0, num_layers-2):
                current_input = X
                W = self.weights[l]
                b = self.biases[l]
                X = current_input + tf.nn.relu(tf.add(tf.matmul(X, W), b))
                #tf.summary.histogram("activation" + str(l+1), X)
            W = self.weights[-1]
            b = self.biases[-1]
            output = tf.add(tf.matmul(X, W), b)
            return output
    
