#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:14:40 2019

@author: hwan
"""

###############################################################################
#                        Generate List of Scenarios                           #
###############################################################################
def get_hyperparameter_permutations(hyperp):
    
    # Create list of hyperparameters and list of scenarios
    hyperp_dict = hyperp.__dict__ # converts all instance attributes into dictionary. Note that it does not include class attributes! In our case, this is GPU
    hyperp_keys = list(hyperp_dict.keys())
    hyperp_dict_list = list(hyperp_dict.values())           
    permutations_list = assemble_permutations(hyperp_dict_list) # list of lists containing all permutations of the hyperparameters

    return permutations_list, hyperp_keys

def assemble_permutations(hyperp_dict_list):
    # params is a list of lists, with each inner list representing
    # a different model parameter. This function constructs the combinations
    return get_combinations(hyperp_dict_list[0], hyperp_dict_list[1:])
    
def get_combinations(hyperp, hyperp_dict_list):
    # assign here in case this is the last list item
    combos = hyperp_dict_list[0]
    
    # reassign when it is not the last item - recursive algorithm
    if len(hyperp_dict_list) > 1:
        combos = get_combinations(hyperp_dict_list[0], hyperp_dict_list[1:])
        
    # concatenate the output into a list of lists
    output = []
    for i in hyperp:
        for j in combos:
            # convert to list if not already
            j = j if isinstance(j, list) else [j]            
            # for some reason, this needs broken into 3 lines...Python
            temp = [i]
            temp.extend(j)
            output.append(temp)
    return output    