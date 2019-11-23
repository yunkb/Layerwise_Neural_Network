#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:41:12 2019

@author: hwan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:55:12 2019

@author: hwan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:13:20 2019

@author: hwan
"""
import tensorflow as tf

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.form_train_val_test_batches import form_train_val_test_batches
from Utilities.NN_FC_layerwise import FCLayerwise
from Utilities.loss_and_accuracies import data_loss_regression, relative_error
from Utilities.optimize_layerwise import optimize

from decimal import Decimal # for filenames

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    num_hidden_nodes  = 50
    regularization    = 0.001
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 30
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyperp):    
        #=== Use L_1 Regularization ===#
        self.use_L1 = 1
        
        #=== Data Set ===#
        data_thermal_fin_nine = 0
        data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_training_data = 50000
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 0
        self.fin_dimensions_3D = 1
        
        #=== Random Seed ===#
        self.random_seed = 1234

###############################################################################
#                                 File Name                                   #
###############################################################################                
        #=== Parameter and Observation Dimensions === #
        self.NN_type = 'FC'
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions
        
        #=== File name ===#
        if data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if self.fin_dimensions_2D == 1:
            fin_dimension = ''
        if self.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if hyperp.regularization >= 1:
            hyperp.regularization = int(hyperp.regularization)
            regularization_string = str(hyperp.regularization)
        else:
            regularization_string = str(hyperp.regularization)
            regularization_string = 'pt' + regularization_string[2:]                        
        node_TOL_string = str('%.2e' %Decimal(hyperp.node_TOL))
        node_TOL_string = node_TOL_string[-1]
        error_TOL_string = str('%.2e' %Decimal(hyperp.error_TOL))
        error_TOL_string = error_TOL_string[-1]
                
        if self.use_L1 == 0:
            self.filename = self.dataset + '_' + hyperp.data_type + fin_dimension + '_' + self.NN_type + '_mhl%d_hl%d_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)
        else:
            self.filename = self.dataset + '_' + hyperp.data_type + fin_dimension + '_' + self.NN_type + '_L1_mhl%d_hl%d_r%s_nTOL%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, regularization_string, node_TOL_string, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)

###############################################################################
#                                 File Paths                                  #
###############################################################################             
        #=== Loading and saving data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyperp.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(self.num_training_data) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(self.num_training_data) + fin_dimension + '_' + hyperp.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) + fin_dimension + '_' + hyperp.data_type + parameter_type

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
    obs_indices, parameter_train, state_obs_train, parameter_test, state_obs_test, data_input_shape, parameter_dimension = load_thermal_fin_data(run_options, run_options.num_training_data) 
    parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, num_training_data, num_batches_train, num_batches_val = form_train_val_test_batches(run_options.num_training_data, parameter_train, state_obs_train, parameter_test, state_obs_test, hyperp.batch_size, run_options.random_seed)
    output_dimensions = len(obs_indices)
        
    #=== Neural network ===#
    if run_options.use_L1 == 0:
        kernel_regularizer = None
        bias_regularizer = None  
    else:
        kernel_regularizer = tf.keras.regularizers.l1(hyperp.regularization)
        bias_regularizer = tf.keras.regularizers.l1(hyperp.regularization)
    NN = FCLayerwise(hyperp, run_options, data_input_shape, output_dimensions,
                     kernel_regularizer, bias_regularizer,
                     run_options.NN_savefile_directory)    
    
    #=== Training ===#
    optimize(hyperp, run_options, NN, data_loss_regression, relative_error, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, output_dimensions, num_batches_train)
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyperp = Hyperparameters()
    
    if len(sys.argv) > 1:
        hyperp.data_type         = str(sys.argv[1])
        hyperp.max_hidden_layers = int(sys.argv[2])
        hyperp.filter_size       = int(sys.argv[3])
        hyperp.num_filters       = int(sys.argv[4])
        hyperp.regularization    = float(sys.argv[5])
        hyperp.node_TOL          = float(sys.argv[6])
        hyperp.error_TOL         = float(sys.argv[7])
        hyperp.num_training_data = int(sys.argv[8])
        hyperp.batch_size        = int(sys.argv[9])
        hyperp.num_epochs        = int(sys.argv[10])
        hyperp.gpu               = int(sys.argv[11])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyperp)
    
    #=== Initiate training ===#
    trainer(hyperp, run_options) 