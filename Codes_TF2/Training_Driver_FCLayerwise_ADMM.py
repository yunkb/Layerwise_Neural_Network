#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:13:20 2019

@author: hwan
"""
from Utilities.get_image_data import load_data
from Utilities.form_train_val_test_batches import form_train_val_test_batches
from Utilities.NN_FC_layerwise import FCLayerwise
from Utilities.loss_and_accuracies import data_loss_classification, accuracy_classification
from Utilities.optimize_ADMM_layerwise import optimize_ADMM

from decimal import Decimal # for filenames

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class Hyperparameters:
    max_hidden_layers = 5 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    num_hidden_nodes  = 100
    activation        = 'elu'
    regularization    = 0.001
    penalty           = 0.001
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 3
    gpu               = '1'
    
class RunOptions:
    def __init__(self):  
        #=== Choose Which GPU to Use ===#
        self.which_gpu = '1'
        
        #=== Choose Data Set ===#
        self.data_MNIST = 1
        self.data_CIFAR10 = 0 
        self.data_CIFAR100 = 0
        
        #=== Random Seed ===#
        self.random_seed = 1234
        
###############################################################################
#                                 File Paths                                  #
###############################################################################  
class FilePaths():    
    def __init__(self, hyperp, run_options):  
        #=== Declaring File Name Components ===# 
        self.NN_type = 'FC'
        if run_options.data_MNIST == 1:
            self.dataset = 'MNIST'
        if run_options.data_CIFAR10 == 1:
            self.dataset = 'CIFAR10'
        if run_options.data_CIFAR100 == 1:
            self.dataset = 'CIFAR100'
        if hyperp.regularization >= 1:
            hyperp.regularization = int(hyperp.regularization)
            regularization_string = str(hyperp.regularization)
        else:
            regularization_string = str(hyperp.regularization)
            regularization_string = 'pt' + regularization_string[2:]            
        if hyperp.penalty >= 1:
            hyperp.penalty = int(hyperp.penalty)
            penalty_string = str(hyperp.penalty)
        else:
            penalty_string = str(hyperp.penalty)
            penalty_string = 'pt' + penalty_string[2:]            
        node_TOL_string = str('%.2e' %Decimal(hyperp.node_TOL))
        node_TOL_string = node_TOL_string[-1]
        error_TOL_string = str('%.2e' %Decimal(hyperp.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        #=== File Name ===#
        self.filename = self.dataset + '_' + self.NN_type + '_ADMM_mhl%d_hl%d_r%s_p%s_nTOL%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, regularization_string, penalty_string, node_TOL_string, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.tensorboard_directory = '../Tensorboard/' + self.filename

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyperp, run_options, file_paths):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    
    #=== Load Data ===#       
    data_train, labels_train,\
    data_test, labels_test,\
    data_input_shape, num_channels, label_dimensions\
    = load_data(file_paths.NN_type, file_paths.dataset, run_options.random_seed) 
    
    #=== Construct Validation Set and Batches ===# 
    data_and_labels_train, data_and_labels_val, data_and_labels_test,\
    num_data_train, num_data_val, num_data_test,\
    num_batches_train, num_batches_val, num_batches_test\
    = form_train_val_test_batches(data_train, labels_train, \
                                  data_test, labels_test, \
                                  hyperp.batch_size, run_options.random_seed)
    
    #=== Neural Network ===#
    NN = FCLayerwise(hyperp, run_options, data_input_shape, label_dimensions,
                      None, None)     
    
    #=== Training ===#
    optimize_ADMM(hyperp, run_options, file_paths, NN, data_loss_classification, accuracy_classification, data_and_labels_train, data_and_labels_val, data_and_labels_test, label_dimensions, num_batches_train)
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters and Run Options ===#    
    hyperp = Hyperparameters()
    run_options = RunOptions()
    
    if len(sys.argv) > 1:
        hyperp.max_hidden_layers = int(sys.argv[1])
        hyperp.num_hidden_nodes  = int(sys.argv[2])
        hyperp.activation        = str(sys.argv[3])
        hyperp.regularization    = float(sys.argv[4])
        hyperp.penalty           = float(sys.argv[5])
        hyperp.node_TOL          = float(sys.argv[6])
        hyperp.error_TOL         = float(sys.argv[7])
        hyperp.batch_size        = int(sys.argv[8])
        hyperp.num_epochs        = int(sys.argv[9])
        run_options.which_gpu    = int(sys.argv[19])
            
    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Initiate training ===#
    trainer(hyperp, run_options, file_paths) 