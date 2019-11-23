#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class CNNLayerwise(tf.keras.Model):
    def __init__(self, hyperp, run_options, data_input_shape, label_dimensions, num_channels, kernel_regularizer, bias_regularizer):
        super(CNNLayerwise, self).__init__()
###############################################################################
#                  Construct Initial Neural Network Architecture               #
###############################################################################
        #=== Defining Attributes ===#
        self.data_input_shape = data_input_shape
        self.architecture = [] # storage for layer information, each entry is [filter_size, num_filters]
        self.num_filters = hyperp.num_filters
        self.kernel_size = hyperp.filter_size
        self.activation = hyperp.activation
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.hidden_layers_list = [] # This will be a list of Keras layers

        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append([self.data_input_shape[0], num_channels]) # input information
        self.architecture.append([3, self.num_filters]) # 3x3 convolutional layer for upsampling data
        self.architecture.append([hyperp.filter_size, self.num_filters]) # First hidden layer
        self.architecture.append([3, num_channels]) # 3x3 convolutional layer for downsampling features
        self.architecture.append(label_dimensions) # fully-connected output layer
        print(self.architecture)
        
        #=== Weights and Biases Initializer ===#
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)
        bias_initializer = 'zeros'
        
        #=== Linear Upsampling Layer to Map to Feature Space ===#
        l = 1
        self.upsampling_layer = Conv2D(self.architecture[l][1], (3, 3), padding = 'same',
                                       activation = 'linear', use_bias = True,
                                       input_shape = self.data_input_shape,
                                       kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                       kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                       name='upsampling_layer')
        
        #=== Define Hidden Layers ===#
        l = 2
        conv_layer = Conv2D(self.architecture[l][1], (self.architecture[l][0], self.architecture[l][0]), padding = 'same', 
                            activation = self.activation, use_bias = True, 
                            input_shape = (None, self.data_input_shape[0], self.data_input_shape[1], self.num_filters),
                            kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                            kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                            name = "W" + str(l))
        self.hidden_layers_list.append(conv_layer)
            
            
        #=== Linear Downsampling Layer to Map to Data Space ===#
        l = 3
        self.downsampling_layer = Conv2D(self.architecture[l][1], (3, 3), padding = 'same',
                                         activation = "linear", use_bias = True,
                                         input_shape = (None, self.data_input_shape[0], self.data_input_shape[1], self.num_filters),
                                         kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                         kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                         name = "downsampling_layer")
        
        #=== Classification Layer ===#
        self.classification_layer = Dense(units = label_dimensions,
                                          activation = 'linear', use_bias = True,
                                          kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                          kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                          name = 'classification_layer')
        
###############################################################################
#                            Network Propagation                              #
############################################################################### 
    def call(self, inputs):
        #=== Upsampling ===#
        output = self.upsampling_layer(inputs)  
        for hidden_layer in self.hidden_layers_list:
            #=== Hidden Layers ===#
            prev_output = output
            output = prev_output + hidden_layer(output)
        #=== Downsampling ===#
        output = self.downsampling_layer(output)
        #=== Classification ===#
        output = Flatten()(output)
        output = self.classification_layer(output)
        return output
    
###############################################################################
#                                 Add Layer                                   #
###############################################################################     
    def add_layer(self, trainable_hidden_layer_index, freeze = True, add = True):
        kernel_initializer = 'zeros'
        bias_initializer = 'zeros'
        if add:
            conv_layer = Conv2D(self.num_filters, (self.kernel_size, self.kernel_size), padding = 'same',
                                activation = self.activation, use_bias = True,
                                input_shape = (None, self.data_input_shape[0], self.data_input_shape[1], self.num_filters),
                                kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                name = "W" + str(trainable_hidden_layer_index))
            self.hidden_layers_list.append(conv_layer)
        if freeze:
            self.upsampling_layer.trainable = False
            for index in range(0, trainable_hidden_layer_index-2):
              self.hidden_layers_list[index].trainable = False
        else:
            self.upsampling_layer.trainable = True
            for index in range(0, trainable_hidden_layer_index-2):
              self.hidden_layers_list[index].trainable = True
              
###############################################################################
#                              Sparsify Weights                               #
###############################################################################            
    def sparsify_weights_and_get_relative_number_of_zeros(self, threshold = 1e-6):
        #=== Downsampling Layer ===#
        down_weights = self.downsampling_layer.get_weights()        
        sparsified_weights = self.sparsify_weights(down_weights, threshold)
        self.downsampling_layer.set_weights(sparsified_weights)
        
        #=== Classification Layer ===#
        class_weights = self.classification_layer.get_weights()
        sparsified_weights = self.sparsify_weights(class_weights, threshold)
        self.classification_layer.set_weights(sparsified_weights)
        
        #=== Trained Hidden Layer ===#
        trained_weights = self.hidden_layers_list[-1].get_weights()
        sparsified_weights = self.sparsify_weights(trained_weights, threshold)
        self.hidden_layers_list[-1].set_weights(sparsified_weights)
        
        #=== Compute Relative Number of Zeros ===#
        total_number_of_zeros = 0
        total_number_of_elements = 0
        for i in range(0, len(sparsified_weights)):
            total_number_of_zeros += np.count_nonzero(sparsified_weights[i]==0)
            total_number_of_elements += sparsified_weights[i].flatten().shape[0]
        relative_number_zeros = np.float64(total_number_of_zeros/total_number_of_elements)
        
        return relative_number_zeros
    
    def sparsify_weights(self, weights, threshold = 1e-6):
        sparsified_weights = []
        if isinstance(weights, float):
            if abs(weights) > threshold:
                sparsified_weights = weights
            else:
                sparsified_weights = 0
        else:
            for w in weights:
                bool_mask = (abs(w) > threshold).astype(int)
                sparsified_weights.append(w*bool_mask)
            
        return sparsified_weights
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    