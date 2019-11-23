#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import subprocess
import copy
from Utilities.get_hyperparameter_permutations import get_hyperparameter_permutations
from plotting_results_CNNLayerwise_ADMM import Hyperparameters
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':                        
    #########################
    #   Get Scenarios List  #
    #########################   
    hyperp = Hyperparameters() # Assign instance attributes below, DO NOT assign an instance attribute to GPU
    
    # assign instance attributes for hyperp
    hyperp.max_hidden_layers = [7]
    hyperp.filter_size       = [3] # Indexing includes input and output layer with input layer indexed by 0
    hyperp.num_filters       = [64]
    hyperp.regularization    = [0.01, 0.1, 1]
    hyperp.penalty           = [0.0001, 0.001, 0.01, 1]
    hyperp.node_TOL          = [1e-4]
    hyperp.error_TOL         = [1e-4]
    hyperp.batch_size        = [1000]
    hyperp.num_epochs        = [30]
    
    permutations_list, hyperp_keys = get_hyperparameter_permutations(hyperp) 
    print('permutations_list generated')
    
    # Convert each list in permutations_list into class attributes
    scenarios_class_instances = []
    for scenario_values in permutations_list: 
        hyperp_scenario = Hyperparameters()
        for i in range(0, len(scenario_values)):
            setattr(hyperp_scenario, hyperp_keys[i], scenario_values[i])
        scenarios_class_instances.append(copy.deepcopy(hyperp_scenario))

    for scenario in scenarios_class_instances:
        subprocess.Popen(['./Plotting_Results_CNNLayerwise_ADMM.py', f'{scenario.max_hidden_layers}', f'{scenario.filter_size}', f'{scenario.num_filters}', f'{scenario.regularization}', f'{scenario.penalty:.4f}', f'{scenario.node_TOL:.4e}', f'{scenario.error_TOL:.4e}', f'{scenario.batch_size}', f'{scenario.num_epochs}',  f'{scenario.gpu}'])
    
    print('All scenarios plotted')