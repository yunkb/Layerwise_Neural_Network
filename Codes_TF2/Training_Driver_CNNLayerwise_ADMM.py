#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:13:20 2019

@author: hwan
"""
from Utilities.get_data import load_data
from Utilities.NN_CNN_layerwise import CNNLayerwise
from Utilities.optimize_ADMM_layerwise import optimize_ADMM

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
    num_filters       = 192
    regularization    = 1
    penalty           = 1e-5
    node_TOL          = 1e-3
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 30
    gpu               = '3'
    
class RunOptions:
    def __init__(self, hyper_p):        
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
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]            
        node_TOL_string = str('%.2e' %Decimal(hyper_p.node_TOL))
        node_TOL_string = node_TOL_string[-1]
        error_TOL_string = str('%.2e' %Decimal(hyper_p.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        self.filename = self.dataset + '_' + self.NN_type + '_ADMM_mhl%d_fs%d_nf%d_r%s_p%s_nTOL%s_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, hyper_p.filter_size, hyper_p.num_filters, regularization_string, penalty_string, node_TOL_string, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

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
    NN = CNNLayerwise(hyper_p, run_options, data_input_shape, label_dimensions, num_channels,
                      None, None,
                      run_options.NN_savefile_directory, construct_flag = 1)    
    
    #=== Training ===#
    optimize_ADMM(hyper_p, run_options, NN, data_and_labels_train, data_and_labels_test, data_and_labels_val, label_dimensions, num_batches_train)
    
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
        hyper_p.penalty           = float(sys.argv[5])
        hyper_p.node_TOL          = float(sys.argv[6])
        hyper_p.error_TOL         = float(sys.argv[7])
        hyper_p.batch_size        = int(sys.argv[8])
        hyper_p.num_epochs        = int(sys.argv[9])
        hyper_p.gpu               = int(sys.argv[10])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 