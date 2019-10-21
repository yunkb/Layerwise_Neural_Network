#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:53:47 2019

@author: hwan
"""

import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def save_weights_and_biases(sess, weight_list_counter, savefilepath):
    #=== Save Newly Trained Weights and Biases ===#
    l = weight_list_counter
    trained_weights = {"W"+str(l+1): sess.run("NN_layerwise/W" + str(l+1) + ':0').flatten()}
    trained_biases = {"b"+str(l+1): sess.run("NN_layerwise/b" + str(l+1) + ':0').flatten()}
    df_trained_weights = pd.DataFrame(trained_weights)
    df_trained_biases = pd.DataFrame(trained_biases)
    df_trained_weights.to_csv(savefilepath + "_W" + str(l+1) + '.csv', index=False)
    df_trained_biases.to_csv(savefilepath + "_b" + str(l+1) + '.csv', index=False)
    
    #=== Save Output Weights and Biases ===# Note that these get replaced everytime
    trained_output_weights = {"Woutput": sess.run("NN_layerwise/W" + str(l+2) + ':0').flatten()}
    trained_output_biases = {"boutput": sess.run("NN_layerwise/b" + str(l+2) + ':0').flatten()}
    df_trained_output_weights = pd.DataFrame(trained_output_weights)
    df_trained_output_biases = pd.DataFrame(trained_output_biases)
    df_trained_output_weights.to_csv(savefilepath + "_Woutput" + '.csv', index=False)
    df_trained_output_biases.to_csv(savefilepath + "_boutput" + '.csv', index=False)
    
# =============================================================================
#     #=== Testing restore ===#
#     df_trained_weights = pd.read_csv(savefilepath + "_W" + str(l+1) + '.csv')
#     df_trained_biases = pd.read_csv(savefilepath + "_b" + str(l+1) + '.csv')
#     restored_W = df_trained_weights.values.reshape([layers[l], layers[l + 1]])
#     restored_b = df_trained_biases.values.reshape([1, layers[l + 1]])
#     pdb.set_trace()
# =============================================================================
