#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:43:31 2019

@author: hwan
"""

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL) # Suppresses all the messages when run begins
import numpy as np
import pandas as pd

from Utilities.NN_FC_layerwise import FullyConnectedLayerwise
from Utilities.get_MNIST_data import load_MNIST_data
from Utilities.get_CIFAR10_data import load_CIFAR10_data
from Utilities.ADMM_methods_FC import construct_ADMM_objects, update_z_and_lagrange_multiplier_tf_operations
from Utilities.optimize_ADMM_layerwise import optimize_ADMM_layerwise

from decimal import Decimal # for filenames

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')


###############################################################################
#                       Hyperparameters and RunOptions                        #
###############################################################################
class HyperParameters:
    max_hidden_layers = 8 
    regularization    = 1
    penalty           = 1
    node_TOL          = 1e-3
    error_TOL         = 1e-2
    batch_size        = 100
    num_epochs        = 60
    gpu               = '1'
    
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
        
        self.filename = data_type + '_' + self.NN_type + '_ADMM_mhl%d_r%s_p%s_nTOL%s_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, regularization_string, penalty_string, node_TOL_string, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

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
        
    #=== Iteration Objects ===#
    loss_value = 1e5
    trainable_hidden_layer_index = 1
    storage_relative_number_zeros_array = np.array([])
    
    while loss_value > hyper_p.error_TOL and trainable_hidden_layer_index < hyper_p.max_hidden_layers:       
        ###########################
        #   Training Properties   #
        ###########################   
        #=== Neural network ===#
        NN = FullyConnectedLayerwise(hyper_p, trainable_hidden_layer_index, data_dimensions, label_dimensions, run_options.NN_savefile_name)
        
        #=== Initialize ADMM objects ===#
        z_weights, z_biases, lagrange_weights, lagrange_biases = construct_ADMM_objects(NN)
        alpha = tf.constant(hyper_p.regularization, dtype = tf.float32)
        pen = tf.constant(hyper_p.penalty, dtype = tf.float32)
        update_z_and_lagrange_multiplier_tf_operations(NN, alpha, pen, z_weights, z_biases, lagrange_weights, lagrange_biases)
        
        #=== Train ===#
        storage_loss_array, storage_accuracy_array, relative_number_zeros = optimize_ADMM_layerwise(hyper_p, run_options, trainable_hidden_layer_index, NN, num_training_data, num_testing_data, pen, z_weights, z_biases, lagrange_weights, lagrange_biases, data_train, labels_train, data_test, labels_test)
        
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss'] = storage_loss_array
        metrics_dict['accuracy'] = storage_accuracy_array
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(run_options.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv', index=False)
        
        #=== Prepare for Next Layer ===#
        tf.reset_default_graph()
        trainable_hidden_layer_index += 1
    
    #=== Saving Relative Number of Zero Elements ===#
    relative_number_zeros_dict = {}
    relative_number_zeros_dict['rel_zeros'] = storage_relative_number_zeros_array
    df_relative_number_zeros = pd.DataFrame(relative_number_zeros_dict)
    df_relative_number_zeros.to_csv(run_options.NN_savefile_name + "_relzeros" + '.csv', index=False)
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
            hyper_p.max_hidden_layers = int(sys.argv[1])
            hyper_p.regularization    = float(sys.argv[2])
            hyper_p.penalty           = float(sys.argv[3])
            hyper_p.node_TOL          = float(sys.argv[4])
            hyper_p.error_TOL         = float(sys.argv[5])
            hyper_p.batch_size        = int(sys.argv[6])
            hyper_p.num_epochs        = int(sys.argv[7])
            hyper_p.gpu               = str(sys.argv[8])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     