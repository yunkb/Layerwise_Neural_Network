#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:38:28 2019

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
    max_hidden_layers = 8 # For this architecture, need at least 2. One for the mapping to the feature space, one as a trainable hidden layer. EXCLUDES MAPPING BACK TO DATA SPACE
    filter_size       = 3
    num_filters       = 128
    regularization    = 0.001
    penalty           = 5
    node_TOL          = 1e-4
    error_TOL         = 1e-4
    batch_size        = 1000
    num_epochs        = 30
    gpu               = '3'
    
class RunOptions:
    def __init__(self, hyperp):        
        #=== Choose Data Set ===#
        data_MNIST = 1
        data_CIFAR10 = 0
        data_CIFAR100 = 0
        
        #=== Unfreeze All Layers and Train ===#
        self.use_unfreeze_all_and_train = 0
        
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

        #=== Saving neural network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we save the parameters for each layer separately, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the saved parameters
        
        #=== Saving Figures ===#
        self.figures_savefile_directory = '../Figures/' + self.filename

        #=== Creating Directories ===#
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
            
###############################################################################
#                                  Driver                                     #
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
        hyperp.gpu               = int(sys.argv[10])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyperp)
    
    #=== Plot and save figures ===#
    plot_and_save_figures(hyperp, run_options)


