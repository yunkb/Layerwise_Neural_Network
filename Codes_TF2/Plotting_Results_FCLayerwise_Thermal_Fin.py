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
class HyperParameters:
    data_type         = 'full'
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    num_hidden_nodes  = 500
    regularization    = 0.001
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 30
    gpu               = '1'
    
class RunOptions:
    def __init__(self, hyper_p):    
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
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
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
            self.filename = self.dataset + '_' + hyper_p.data_type + fin_dimension + '_' + self.NN_type + '_L1_mhl%d_hl%d_r%s_nTOL%s_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, hyper_p.num_hidden_nodes, regularization_string, node_TOL_string, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
###############################################################################                     
        #=== Saving neural network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we save the parameters for each layer separately, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the saved parameters

        #=== Saving Figures ===#
        self.figures_savefile_directory = '../Figures/' + self.filename

        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
            
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
        hyper_p.max_hidden_layers = int(sys.argv[1])
        hyper_p.filter_size       = int(sys.argv[2])
        hyper_p.num_filters       = int(sys.argv[3])
        hyper_p.regularization    = float(sys.argv[4])
        hyper_p.reg_schedule      = float(sys.argv[5])
        hyper_p.node_TOL          = float(sys.argv[6])
        hyper_p.error_TOL         = float(sys.argv[7])
        hyper_p.batch_size        = int(sys.argv[8])
        hyper_p.num_epochs        = int(sys.argv[9])
        hyper_p.gpu               = int(sys.argv[10])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Plot and save figures ===#
    plot_and_save_figures(hyper_p, run_options)
