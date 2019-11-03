#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:13:20 2019

@author: hwan
"""
import tensorflow as tf

from Utilities.get_data import load_data
from Utilities.NN_CNN_layerwise import CNNLayerwise
from Utilities.optimize_layerwise import optimize

from decimal import Decimal # for filenames

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    filter_size       = 3
    num_filters       = 64
    regularization    = 1
    node_TOL          = 1e-4
    error_TOL         = 1e-2
    batch_size        = 1000
    num_epochs        = 30
    gpu               = '2'
    
class RunOptions:
    def __init__(self, hyper_p):    
        #=== Use L_1 Regularization ===#
        self.use_L1 = 1
        
        #=== Choose Data Set ===#
        data_MNIST = 0
        data_CIFAR10 = 1
        data_CIFAR100 = 0
        
        #=== Random Seed ===#
        self.random_seed = 1234
        
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
        
        #=== Setting Filename ===# 
        self.NN_type = 'CNN'
        if data_MNIST == 1:
            self.dataset = 'MNIST'
        if data_CIFAR10 == 1:
            self.dataset = 'CIFAR10'
        if data_CIFAR100 == 1:
            self.dataset = 'CIFAR100'
        if hyper_p.regularization >= 1:
            hyper_p.regularization = int(hyper_p.regularization)
            regularization_string = str(hyper_p.regularization)
        else:
            regularization_string = str(hyper_p.regularization)
            regularization_string = 'pt' + regularization_string[2:]                        
        node_TOL_string = str('%.2e' %Decimal(hyper_p.node_TOL))
        node_TOL_string = node_TOL_string[-1]
        error_TOL_string = str('%.2e' %Decimal(hyper_p.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        if self.use_L1 == 0:
            self.filename = self.dataset + '_' + self.NN_type + '_mhl%d_fs%d_nf%d_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, hyper_p.filter_size, hyper_p.num_filters, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)
        else:
            self.filename = self.dataset + '_' + self.NN_type + '_L1_mhl%d_fs%d_nf%d_r%s_nTOL%s_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, hyper_p.filter_size, hyper_p.num_filters, regularization_string, node_TOL_string, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Saving neural network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we save the parameters for each layer separately, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the saved parameters

        #=== Creating Directories ===#
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyper_p, run_options):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = hyper_p.gpu
    
    #=== Load Train and Test Data ===#  
    data_and_labels_train, data_and_labels_test, data_and_labels_val, data_input_shape, num_channels, label_dimensions, num_batches_train, num_batches_val = load_data(run_options.dataset, hyper_p.batch_size, run_options.random_seed)  
    
    #=== Neural network ===#
    if run_options.use_L1 == 0:
        kernel_regularizer = None
        bias_regularizer = None  
    else:
        kernel_regularizer = tf.keras.regularizers.l1(hyper_p.regularization)
        bias_regularizer = tf.keras.regularizers.l1(hyper_p.regularization)
    NN = CNNLayerwise(hyper_p, run_options, data_input_shape, label_dimensions, num_channels,
                      kernel_regularizer, bias_regularizer,
                      run_options.NN_savefile_directory, construct_flag = 1)    
    
    #=== Training ===#
    optimize(hyper_p, run_options, NN, data_and_labels_train, data_and_labels_test, data_and_labels_val, label_dimensions, num_batches_train)
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
        hyper_p.max_hidden_layers = int(sys.argv[1])
        hyper_p.filter_size       = int(sys.argv[2])
        hyper_p.num_filters       = int(sys.argv[3])
        hyper_p.regularization    = float(sys.argv[4])
        hyper_p.node_TOL          = float(sys.argv[5])
        hyper_p.error_TOL         = float(sys.argv[6])
        hyper_p.batch_size        = int(sys.argv[7])
        hyper_p.num_epochs        = int(sys.argv[8])
        hyper_p.gpu               = int(sys.argv[9])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 