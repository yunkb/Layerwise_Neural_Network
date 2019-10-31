#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 20:03:31 2019

@author: hwan
"""

from plot_and_save_figures_layerwise import plot_and_save_figures
from decimal import Decimal # for filenames
import os
import sys

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    max_hidden_layers = 8
    error_TOL         = 1e-2
    batch_size        = 100
    num_epochs        = 15
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p, data_type):   
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
                
        #=== Setting Filename ===# 
        self.NN_type = 'FC'
        
        #=== Filename ===#
        error_TOL_string = str('%.2e' %Decimal(hyper_p.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        self.filename = data_type + '_' + self.NN_type + '_mhl%d_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Saving Neural Network ===#
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
    
    #=== Set hyperparameters ===#
    hyper_p = HyperParameters()
        
    #=== Set run options ===#     
    if len(sys.argv) > 1:
            data_type = str(sys.argv[1])   
    else:
        data_type = 'MNIST'
            
    run_options = RunOptions(hyper_p, data_type)
    
    #=== Plot and save figures ===#
    plot_and_save_figures(hyper_p, run_options)


        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            