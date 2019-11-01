#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class CNNLayerwise(tf.keras.Model):
    def __init__(self, hyper_p, run_options, data_input_shape, label_dimensions, num_channels, kernel_regularizer, bias_regularizer, savefilepath, construct_flag):
        super(CNNLayerwise, self).__init__()
###############################################################################
#                  Constuct Initial Neural Network Architecture               #
###############################################################################
        #=== Defining Attributes ===#
        self.data_input_shape = data_input_shape
        self.architecture = [] # storage for layer information, each entry is [filter_size, num_filters]
        self.num_filters = hyper_p.num_filters
        self.kernel_size = hyper_p.filter_size
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.hidden_layers_list = [] # This will be a list of Keras layers

        #=== Define Initial Architecture and Create Layer Storage ===#
        self.architecture.append([self.data_input_shape[0], num_channels]) # input information
        self.architecture.append([1, self.num_filters]) # 1x1 convolutional layer for upsampling data
        self.architecture.append([hyper_p.filter_size, self.num_filters]) # First hidden layer
        self.architecture.append([1, num_channels]) # 1x1 convolutional layer for downsampling features
        self.architecture.append(label_dimensions) # fully-connected output layer
        print(self.architecture)

        #=== Weights and Biases Initializer ===#
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)
        bias_initializer = 'zeros'
        
        #=== Linear Upsampling Layer to Map to Feature Space ===#
        l = 1
        self.upsampling_layer = Conv2D(self.architecture[l][1], (1, 1), padding = 'same',
                                       activation = 'linear', use_bias = True,
                                       input_shape = self.data_input_shape,
                                       kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                       kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                       name='upsampling_layer')
        
        #=== Define Hidden Layers ===#
        l = 2
        activation_type = 'relu'
        conv_layer = Conv2D(self.architecture[l][1], (self.architecture[l][0], self.architecture[l][0]), padding = 'same', 
                            activation = activation_type, use_bias = True, 
                            input_shape = (None, self.data_input_shape[0], self.data_input_shape[1], self.num_filters),
                            kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                            kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                            name = "W" + str(l))
        self.hidden_layers_list.append(conv_layer)
            
            
        #=== Linear Downsampling Layer to Map to Data Space ===#
        l = 3
        self.downsampling_layer = Conv2D(self.architecture[l][1], (1, 1), padding = 'same',
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
                                activation ='relu', use_bias = True,
                                input_shape = (None, self.data_input_shape[0], self.data_input_shape[1], self.num_filters),
                                kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                                kernel_regularizer = self.kernel_regularizer, bias_regularizer = self.bias_regularizer,
                                name = "W" + str(trainable_hidden_layer_index))
        self.hidden_layers_list.append(conv_layer)
        if freeze:
            self.upsampling_layer.trainable = False
            for index in range(1, trainable_hidden_layer_index-1):
              self.hidden_layers_list[index].trainable = False
        else:
            self.upsampling_layer.trainable = True
            for index in range(1, trainable_hidden_layer_index-1):
              self.hidden_layers_list[index].trainable = True
              
###############################################################################
#                              Sparsify Weights                               #
###############################################################################            
    def sparsify_weights(self, threshold = 1e-6):
        trained_weights = self.hidden_layers_list[-1].get_weights()
        sparsified_weights = []
        for w in trained_weights:
            bool_mask = (w > threshold).astype(int)
            sparsified_weights.append(w*bool_mask)
        self.hidden_layers_list[-1].set_weights(sparsified_weights)
        
        total_number_of_zeros = 0
        total_number_of_elements = 0
        for i in range(0, len(trained_weights)):
            total_number_of_zeros += np.count_nonzero(trained_weights[i]==0)
            total_number_of_elements += trained_weights[i].flatten().shape[0]
        relative_number_zeros = total_number_of_zeros/total_number_of_elements
        
        return relative_number_zeros
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    