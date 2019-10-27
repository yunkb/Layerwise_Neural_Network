#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:43:31 2019

@author: hwan
"""

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR) # Suppresses all the messages when run begins
import numpy as np
import pandas as pd

from NN_FC_layerwise import FullyConnectedLayerwise
from get_MNIST_data import load_MNIST_data
from get_CIFAR10_data import load_CIFAR10_data
from optimize_L2_layerwise import optimize_L2_layerwise

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

from decimal import Decimal # for filenames
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
    max_hidden_layers = 8
    error_TOL         = 1e-2
    batch_size        = 1000
    num_epochs        = 60
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p):   
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
        
        #=== Choose Data Set ===#
        self.data_MNIST = 0
        self.data_CIFAR10 = 1   
        
        #=== Setting Filename ===# 
        self.NN_type = 'FC'
        if self.data_MNIST == 1:
            data_type = 'MNIST'
        if self.data_CIFAR10 == 1:
            data_type = 'CIFAR10'
        
        #=== Filename ===#
        error_TOL_string = str('%.2e' %Decimal(hyper_p.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        self.filename = data_type + '_' + self.NN_type + '_L2_mhl%d_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

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
    if run_options.data_MNIST == 1:
        num_training_data, num_testing_data, img_size, num_channels, data_dimensions, label_dimensions, data_train, labels_train, data_test, labels_test  = load_MNIST_data()
    if run_options.data_CIFAR10 == 1:    
        num_training_data, num_testing_data, img_size, num_channels, data_dimensions, label_dimensions, class_names, data_train, labels_train, data_test, labels_test = load_CIFAR10_data()
        
    loss_value = 1e5
    trainable_hidden_layer_index = 1

    while loss_value > hyper_p.error_TOL and trainable_hidden_layer_index < hyper_p.max_hidden_layers:    
        #=== Neural network ===#
        NN = FullyConnectedLayerwise(hyper_p, trainable_hidden_layer_index, data_dimensions, label_dimensions, run_options.NN_savefile_name)
        
        #=== Train ===#
        storage_loss_array, storage_accuracy_array = optimize_L2_layerwise(hyper_p, run_options, trainable_hidden_layer_index, NN, num_training_data, num_testing_data, data_train, labels_train, data_test, labels_test)   
        
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss'] = storage_loss_array
        metrics_dict['accuracy'] = storage_accuracy_array
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(run_options.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv', index=False)
        
        #=== Prepare for Next Layer ===#
        tf.reset_default_graph()
        trainable_hidden_layer_index += 1
        
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
            hyper_p.max_hidden_layers = int(sys.argv[1])
            hyper_p.error_TOL         = float(sys.argv[2])
            hyper_p.batch_size        = int(sys.argv[3])
            hyper_p.num_epochs        = int(sys.argv[4])
            hyper_p.gpu               = str(sys.argv[5])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     