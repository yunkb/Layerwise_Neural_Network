#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
@adapted from: Magnus Erik Hvass Pedersen: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class ConvolutionalLayerwise:
    def __init__(self, hyper_p, run_options, hidden_layer_counter, data_dimension, label_dimensions, img_size, num_channels, savefilepath):
        
###############################################################################
#                    Constuct Neural Network Architecture                     #
###############################################################################        
        #=== Create placeholders ===#
        self.data_tf = tf.placeholder(tf.float32, shape=[None, data_dimension], name='x')
        self.data_image_tf = tf.reshape(self.data_tf, [-1, img_size, img_size, num_channels])
        self.labels_tf = tf.placeholder(tf.float32, shape=[None, label_dimensions], name='y_true')
                           
        #=== Define Architecture and Create Parameter Storage ===#
        self.layers = [] # storage for layer information, each entry is [filter_size, num_filters]
        self.layers.append([img_size, num_channels]) # input information
        for l in range(1, hidden_layer_counter+1):
            self.layers.append([hyper_p.filter_size, hyper_p.num_filters])
        self.layers.append(label_dimensions) # fully-connected output layer
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)      
        
        ############################
        #   Initial Architecture   #
        ############################        
        # If first iteration, initialize output layer
        if hidden_layer_counter == 2: 
            with tf.variable_scope("NN") as scope:   
                # Convolutional mapping to feature space and first convolutional layer, shape = [filter_size, filter_size, num_input_channels, num_filters]. This format is determined by the TensorFlow API.
                for l in range(1, 3): 
                    W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l][0], self.layers[l][0], self.layers[l-1][1], self.layers[l][1]], initializer = tf.random_normal_initializer())
                    b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [self.layers[l][1]], initializer = tf.constant_initializer(0))                                  
                    tf.summary.histogram("weights" + str(l), W)
                    tf.summary.histogram("biases" + str(l), b)
                    self.weights.append(W)
                    self.biases.append(b)
                
                # Fully Connected Output Layer  
                self.X_flat, self.num_features = self.forward_convolutional_prop(self.data_image_tf, num_layers)  
                l = 3
                W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.num_features, self.layers[l]], initializer = tf.random_normal_initializer())
                b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(0))                                  
                tf.summary.histogram("weights" + str(l), W)
                tf.summary.histogram("biases" + str(l), b)
                self.weights.append(W)
                self.biases.append(b)
                
        ##############################
        #   Extending Architecture   #
        ############################## 
        if hidden_layer_counter > 2: 
            with tf.variable_scope("NN") as scope: 
                # Load convolutional mapping to feature space. Note that these will be trained again
                l = 1
                df_trained_weights = pd.read_csv(savefilepath + "_Winput" + '.csv')
                df_trained_biases = pd.read_csv(savefilepath + "_binput" + '.csv')
                restored_W = df_trained_weights.values.reshape([self.layers[l][0], self.layers[l][0], self.layers[l-1][1], self.layers[l][1]])
                restored_b = df_trained_biases.values.reshape([self.layers[l][1]])
                W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l][0], self.layers[l][0], self.layers[l-1][1], self.layers[l][1]], initializer = tf.constant_initializer(restored_W))
                b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [self.layers[l][1]], initializer = tf.constant_initializer(restored_b))                                  
                self.weights.append(W)
                self.biases.append(b)
                
                # Load pre-trained hidden layer weights and biases
                for l in range(2, hidden_layer_counter):
                    df_trained_weights = pd.read_csv(savefilepath + "_W" + str(l) + '.csv')
                    df_trained_biases = pd.read_csv(savefilepath + "_b" + str(l) + '.csv')
                    restored_W = df_trained_weights.values.reshape([self.layers[l][0], self.layers[l][0], self.layers[l-1][1], self.layers[l][1]])
                    restored_b = df_trained_biases.values.reshape([self.layers[l][1]])
                    W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l][0], self.layers[l][0], self.layers[l-1][1], self.layers[l][1]], initializer = tf.constant_initializer(restored_W), trainable = False)
                    b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [self.layers[l][1]], initializer = tf.constant_initializer(restored_b), trainable = False)                                  
                    self.weights.append(W)
                    self.biases.append(b)
                    
                # Construct new hidden layer and set initial weights and biases as 0  
                l = hidden_layer_counter
                W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l][0], self.layers[l][0], self.layers[l-1][1], self.layers[l][1]], initializer = tf.constant_initializer(0))
                b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [self.layers[l][1]], initializer = tf.constant_initializer(0))                                  
                tf.summary.histogram("weights" + str(l), W)
                tf.summary.histogram("biases" + str(l), b)
                self.weights.append(W)
                self.biases.append(b)
                
                # Load pre-trained output layer weights and biases. Note that these will be trained again
                self.X_flat, self.num_features = self.forward_convolutional_prop(self.data_image_tf, num_layers)  
                l = hidden_layer_counter + 1
                df_trained_weights = pd.read_csv(savefilepath + "_Woutput" + '.csv')
                df_trained_biases = pd.read_csv(savefilepath + "_boutput" + '.csv')
                restored_W = df_trained_weights.values.reshape([self.num_features, self.layers[l]])
                restored_b = df_trained_biases.values.reshape([1, self.layers[l]])
                W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.num_features, self.layers[l]], initializer = tf.constant_initializer(restored_W))
                b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(restored_b))                                  
                self.weights.append(W)
                self.biases.append(b)
                
###############################################################################
#                           Network Propagation                               #
###############################################################################                  
        self.prediction = self.fully_connected_classifier(self.X_flat)     

###############################################################################
#                                Methods                                      #
############################################################################### 
    def forward_convolutional_prop(self, X, num_layers): 
        # Convolutional Hidden Layers 
        for l in range(0, num_layers-2):
            current_input = X
            # Create the TensorFlow operation for convolution.
            # Note the strides are set to 1 in all dimensions.
            # The first and last stride must always be 1,
            # because the first is for the image-number and
            # the last is for the input-channel.
            # But e.g. strides=[1, 2, 2, 1] would mean that the filter
            # is moved 2 pixels across the x- and y-axis of the image.
            # The padding is set to 'SAME' which means the input image
            # is padded with zeroes so the size of the output is the same.
            X = tf.nn.conv2d(input   = X,
                             filter  = self.weights[l],
                             strides = [1, 1, 1, 1],
                             padding = 'SAME')
            if l == 0: # Linear mapping to feature space
                X = X + self.biases[l]
            else:
                X = current_input + tf.nn.relu(X + self.biases[l])    
        # Fully Connected Output Layer 
        X_flat, num_features = self.flatten_layer(X)

        return X_flat, num_features
    
    def flatten_layer(self, layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]
    
        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()
        
        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])
    
        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
    
        # Return both the flattened layer and the number of features.
        return layer_flat, num_features
    
    def fully_connected_classifier(self, X_flat):
        W = self.weights[-1]
        b = self.biases[-1]
        output = tf.add(tf.matmul(X_flat, W), b)
        
        return output