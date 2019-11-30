#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:23:46 2019

@author: hwan
"""

from Utilities.plot_and_save_figures_layerwise import plot_and_save_figures
from decimal import Decimal # for filenames
import os
import sys

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    num_hidden_nodes  = 500
    activation        = 'elu'
    regularization    = 0.001
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 30
    
class RunOptions:
    def __init__(self):    
        #=== Choose Which GPU to Use ===#
        self.which_gpu = '3'
        
        #=== Use L_1 Regularization ===#
        self.use_L1 = 1
        
        #=== Data Set ===#
        self.data_thermal_fin_nine = 0
        self.data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_data_train = 50000
        self.num_data_test = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 0
        self.fin_dimensions_3D = 1
        
        #=== Random Seed ===#
        self.random_seed = 1234
        
        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if self.data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if self.data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions

###############################################################################
#                                 File Paths                                  #
###############################################################################                 
class FilePaths():    
    def __init__(self, hyperp, run_options): 
        #=== Declaring File Name Components ===#
        self.NN_type = 'FC'
        if run_options.data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if run_options.data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if run_options.fin_dimensions_2D == 1:
            fin_dimension = ''
        if run_options.fin_dimensions_3D == 1:
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
        
        #=== File Name ===#        
        if run_options.use_L1 == 0:
            self.filename = self.dataset + '_' + hyperp.data_type + fin_dimension + '_' + self.NN_type + '_mhl%d_hl%d_%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, hyperp.activation, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)
        else:
            self.filename = self.dataset + '_' + hyperp.data_type + fin_dimension + '_' + self.NN_type + '_L1_mhl%d_hl%d_%s_r%s_nTOL%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, hyperp.activation, regularization_string, node_TOL_string, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)
                  
        #=== Save File Directory ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
  
        #=== Save File Path ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyperp.data_type + fin_dimension
    
        #=== Save File Path for One Instance of Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/parameter_test' + fin_dimension
        if hyperp.data_type == 'full':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test' + fin_dimension
        if hyperp.data_type == 'bndonly':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test_bnd' + fin_dimension           
            
        #=== Loading Predictions ===#    
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred' + fin_dimension
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred' + fin_dimension
            
        #=== Savefile Path for Figures ===#    
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test' + fin_dimension
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test' + fin_dimension
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred' + fin_dimension
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred' + fin_dimension
        
        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
            
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    #=== Hyperparameters and Run Options ===#    
    hyperp = Hyperparameters()
    run_options = RunOptions()
    
    if len(sys.argv) > 1:
        hyperp.data_type         = str(sys.argv[1])
        hyperp.max_hidden_layers = int(sys.argv[2])
        hyperp.filter_size       = int(sys.argv[3])
        hyperp.num_filters       = int(sys.argv[4])
        hyperp.activation        = str(sys.argv[5])
        hyperp.regularization    = float(sys.argv[6])
        hyperp.node_TOL          = float(sys.argv[7])
        hyperp.error_TOL         = float(sys.argv[8])
        hyperp.num_training_data = int(sys.argv[9])
        hyperp.batch_size        = int(sys.argv[10])
        hyperp.num_epochs        = int(sys.argv[11])
            
    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Plot and save figures ===#
    plot_and_save_figures(hyperp, run_options, file_paths)
