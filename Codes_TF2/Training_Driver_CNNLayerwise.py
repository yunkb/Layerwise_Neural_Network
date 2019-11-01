#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:13:20 2019

@author: hwan
"""
import numpy as np
import pandas as pd

from Utilities.get_data import load_data
from Utilities.NN_CNN_Layerwise import CNNLayerwise
from Utilities.optimize import optimize

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    filter_size       = 3
    num_filters       = 64
    error_TOL         = 1e-2
    batch_size        = 100
    num_epochs        = 15
    gpu               = '2'
    
class RunOptions:
    def __init__(self, hyper_p):        
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
        
        self.filename = self.dataset + '_' + self.NN_type + '_hl%d_fs%d_nf%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.filter_size, hyper_p.num_filters, hyper_p.batch_size, hyper_p.num_epochs)

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
    #=== Load Train and Test Data ===#  
    data_and_labels_train, data_and_labels_test, data_and_labels_val, data_input_shape, num_channels, label_dimensions, num_batches_train, num_batches_val = load_data(run_options.dataset, hyper_p.batch_size, run_options.random_seed)  
    
    #=== Neural network ===#
    NN = CNNLayerwise(hyper_p, run_options, data_input_shape, label_dimensions, num_channels,
             None, None,
             run_options.NN_savefile_directory, construct_flag = 1)    
    
    #=== Training ===#
    storage_loss_array, storage_accuracy_array = optimize(hyper_p, run_options, NN, data_and_labels_train, data_and_labels_test, data_and_labels_val, label_dimensions, num_batches_train)

    #=== Saving Metrics ===#
    metrics_dict = {}
    metrics_dict['loss'] = storage_loss_array
    metrics_dict['accuracy'] = storage_accuracy_array
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(run_options.NN_savefile_name + "_metrics" + '.csv', index=False)
    
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
        hyper_p.error_TOL         = float(sys.argv[4])
        hyper_p.batch_size        = int(sys.argv[5])
        hyper_p.num_epochs        = int(sys.argv[6])
        hyper_p.gpu               = str(sys.argv[7])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 