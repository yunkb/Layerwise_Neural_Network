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
class HyperParameters:
    data_type         = 'full'
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    num_hidden_nodes  = 500
    regularization    = 0.0002
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 30
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p):    
        #=== Use L_1 Regularization ===#
        self.use_L1 = 1
        
        #=== Data Set ===#
        data_thermal_fin_nine = 1
        data_thermal_fin_vary = 0
        self.num_training_data = 50000
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 1
        self.fin_dimensions_3D = 0
        
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
            self.filename = self.dataset + '_' + hyper_p.data_type + fin_dimension + '_' + self.NN_type + '_mhl%d_hl%d_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, hyper_p.num_hidden_nodes, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)
        else:
            self.filename = self.dataset + '_' + self.NN_type + '_L1_mhl%d_hl%d_r%s_nTOL%s_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, hyper_p.num_hidden_nodes, regularization_string, node_TOL_string, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
###############################################################################             
        #=== Loading and saving data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyper_p.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(self.num_training_data) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(self.num_training_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type

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
    obs_indices, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val = load_thermal_fin_data(run_options, run_options.num_training_data, hyper_p.batch_size, run_options.random_seed) 
    output_dimensions = len(obs_indices)
        
    #=== Neural network ===#
    if run_options.use_L1 == 0:
        kernel_regularizer = None
        bias_regularizer = None  
    else:
        kernel_regularizer = tf.keras.regularizers.l1(hyper_p.regularization)
        bias_regularizer = tf.keras.regularizers.l1(hyper_p.regularization)
    NN = FCLayerwise(hyper_p, run_options, data_input_shape, output_dimensions,
                     kernel_regularizer, bias_regularizer,
                     run_options.NN_savefile_directory)    
    
    #=== Training ===#
    optimize(hyper_p, run_options, NN, data_loss_regression, relative_error, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, output_dimensions, num_batches_train)
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
        hyper_p.data_type         = str(sys.argv[1])
        hyper_p.max_hidden_layers = int(sys.argv[2])
        hyper_p.filter_size       = int(sys.argv[3])
        hyper_p.num_filters       = int(sys.argv[4])
        hyper_p.regularization    = float(sys.argv[5])
        hyper_p.node_TOL          = float(sys.argv[6])
        hyper_p.error_TOL         = float(sys.argv[7])
        hyper_p.num_training_data = int(sys.argv[8])
        hyper_p.batch_size        = int(sys.argv[9])
        hyper_p.num_epochs        = int(sys.argv[10])
        hyper_p.gpu               = int(sys.argv[11])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 