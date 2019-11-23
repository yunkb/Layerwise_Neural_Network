#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:13:20 2019

@author: hwan
"""
from Utilities.get_image_data import load_data
from Utilities.form_train_val_test_batches import form_train_val_test_batches
from Utilities.NN_CNN_layerwise import CNNLayerwise
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
    max_hidden_layers = 3 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    filter_size       = 3
    num_filters       = 12
    regularization    = 0.001
    penalty           = 5
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 1
    gpu               = '1'
    
class RunOptions:
    def __init__(self, hyperp):        
        #=== Choose Data Set ===#
        data_MNIST = 1
        data_CIFAR10 = 0
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
        
        self.filename = self.dataset + '_' + self.NN_type + '_ADMM_mhl%d_fs%d_nf%d_r%s_p%s_nTOL%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.filter_size, hyperp.num_filters, regularization_string, penalty_string, node_TOL_string, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.tensorboard_directory = '../Tensorboard/' + self.filename

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyperp, run_options):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = hyperp.gpu
    
    #=== Loading Data and Constructing Batches ===#        
    data_train, labels_train, data_test, labels_test, data_input_shape, num_channels, label_dimensions = load_data(run_options.NN_type, run_options.dataset, hyperp.batch_size, run_options.random_seed) 
    data_and_labels_train, data_and_labels_test, data_and_labels_val, num_training_data, num_batches_train, num_batches_val = form_train_val_test_batches(len(data_train), data_train, labels_train, data_test, labels_test, hyperp.batch_size, run_options.random_seed)
    
    #=== Neural network ===#
    NN = CNNLayerwise(hyperp, run_options, data_input_shape, label_dimensions, num_channels,
                      None, None,
                      run_options.NN_savefile_directory)    
    
    #=== Training ===#
    optimize_ADMM(hyperp, run_options, NN, data_loss_classification, accuracy_classification, data_and_labels_train, data_and_labels_test, data_and_labels_val, label_dimensions, num_batches_train)
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyperp = Hyperparameters()
    
    if len(sys.argv) > 1:
        hyperp.max_hidden_layers = int(sys.argv[1])
        hyperp.filter_size       = int(sys.argv[2])
        hyperp.num_filters       = int(sys.argv[3])
        hyperp.regularization    = float(sys.argv[4])
        hyperp.penalty           = float(sys.argv[5])
        hyperp.node_TOL          = float(sys.argv[6])
        hyperp.error_TOL         = float(sys.argv[7])
        hyperp.batch_size        = int(sys.argv[8])
        hyperp.num_epochs        = int(sys.argv[9])
        hyperp.gpu               = str(sys.argv[10])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyperp)
    
    #=== Initiate training ===#
    trainer(hyperp, run_options) 