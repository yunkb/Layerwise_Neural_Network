#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import subprocess
import copy
from Utilities.get_hyperparameter_permutations import get_hyperparameter_permutations
from plotting_results_CNNLayerwise_ADMM import HyperParameters
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':                        
    #########################
    #   Get Scenarios List  #
    #########################   
    hyper_p = HyperParameters() # Assign instance attributes below, DO NOT assign an instance attribute to GPU
    
    # assign instance attributes for hyper_p
    hyper_p.max_hidden_layers = [7]
    hyper_p.filter_size       = [3] # Indexing includes input and output layer with input layer indexed by 0
    hyper_p.num_filters       = [64]
    hyper_p.regularization    = [0.01, 0.1, 1]
    hyper_p.penalty           = [0.0001, 0.001, 0.01, 1]
    hyper_p.node_TOL          = [1e-4]
    hyper_p.error_TOL         = [1e-4]
    hyper_p.batch_size        = [1000]
    hyper_p.num_epochs        = [30]
    
    permutations_list, hyper_p_keys = get_hyperparameter_permutations(hyper_p) 
    print('permutations_list generated')
    
    # Convert each list in permutations_list into class attributes
    scenarios_class_instances = []
    for scenario_values in permutations_list: 
        hyper_p_scenario = HyperParameters()
        for i in range(0, len(scenario_values)):
            setattr(hyper_p_scenario, hyper_p_keys[i], scenario_values[i])
        scenarios_class_instances.append(copy.deepcopy(hyper_p_scenario))

    for scenario in scenarios_class_instances:
        subprocess.Popen(['./plotting_results_CNNLayerwise_ADMM.py', f'{scenario.max_hidden_layers}', f'{scenario.filter_size}', f'{scenario.num_filters}', f'{scenario.regularization}', f'{scenario.penalty:.4f}', f'{scenario.node_TOL:.4e}', f'{scenario.error_TOL:.4e}', f'{scenario.batch_size}', f'{scenario.num_epochs}',  f'{scenario.gpu}'])
    
    print('All scenarios plotted')