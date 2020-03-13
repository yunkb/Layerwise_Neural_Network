#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:18:38 2019

@author: hwan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:59:59 2019

@author: hwan
"""
import sys
import os
from decimal import Decimal # for filenames

from Utilities.plot_and_save_predictions_paraview import plot_and_save_predictions_paraview

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
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
        #=== Use ResNet ===#
        self.use_resnet = 0
        
        #=== Use Regularization ===#
        self.use_L1 = 1
        
        #=== Mapping Type ===#
        self.forward_mapping = 0
        self.inverse_mapping = 1
        
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
        if run_options.forward_mapping == 1:
            mapping_type = '_fwd'
        if run_options.inverse_mapping == 1:
            mapping_type = '_inv'
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
            self.filename = self.dataset + mapping_type + '_' + hyperp.data_type + fin_dimension + '_' + self.NN_type + '_mhl%d_hl%d_%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, hyperp.activation, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)
        else:
            self.filename = self.dataset + mapping_type + '_' + hyperp.data_type + fin_dimension + '_' + self.NN_type + '_L1_mhl%d_hl%d_%s_r%s_nTOL%s_eTOL%s_b%d_e%d' %(hyperp.max_hidden_layers, hyperp.num_hidden_nodes, hyperp.activation, regularization_string, node_TOL_string, error_TOL_string, hyperp.batch_size, hyperp.num_epochs)
            
        #=== Savefile Path for Figures ===#    
        self.figures_savefile_directory = '/home/hwan/Documents/Github_Codes/Layerwise_Neural_Network/Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test' + fin_dimension + parameter_type
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test' + fin_dimension + parameter_type
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred'
        
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
        hyperp.num_hidden_layers = int(sys.argv[2])
        hyperp.num_hidden_nodes  = int(sys.argv[3])
        hyperp.activation        = str(sys.argv[4])
        hyperp.regularization    = float(sys.argv[5])
        hyperp.node_TOL          = float(sys.argv[6])
        hyperp.batch_size        = int(sys.argv[7])
        hyperp.num_epochs        = int(sys.argv[8])

    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Plot and Save ===#
    fig_size = (5,5)
    #plot_and_save_predictions(hyperp, run_options, file_paths, fig_size)
    
    #=== Plot and Save Paraview ===#
    if run_options.data_thermal_fin_nine == 1:
        cbar_RGB_parameter_test = [0.5035549, 0.231373, 0.298039, 0.752941, 1.3869196499999998, 0.865003, 0.865003, 0.865003, 2.2702843999999995, 0.705882, 0.0156863, 0.14902]
        cbar_RGB_state_test = [0.0018624861168711026, 0.231373, 0.298039, 0.752941, 0.7141493372496077, 0.865003, 0.865003, 0.865003, 1.4264361883823442, 0.705882, 0.0156863, 0.14902]
    if run_options.data_thermal_fin_vary == 1:
        cbar_RGB_parameter_test = [0.30348012, 0.231373, 0.298039, 0.752941, 1.88775191, 0.865003, 0.865003, 0.865003, 3.4720237000000003, 0.705882, 0.0156863, 0.14902]
        cbar_RGB_state_test = [0.004351256241582283, 0.231373, 0.298039, 0.752941, 0.5831443090996347, 0.865003, 0.865003, 0.865003, 1.1619373619576872, 0.705882, 0.0156863, 0.14902]
    
    plot_and_save_predictions_paraview(run_options, file_paths, cbar_RGB_parameter_test, cbar_RGB_state_test)


    
    
    
    
    
    