#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:53:47 2019

@author: hwan
"""

import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def save_weights_and_biases(sess, hyper_p, weight_list_counter, savefilepath, thresholding_flag):
    #=== Save Newly Trained Weights and Biases ===#
    l = weight_list_counter
    trained_weights = sess.run("NN/W" + str(l+1) + ':0')
    trained_biases = sess.run("NN/b" + str(l+1) + ':0')
    if thresholding_flag == 1:
        trained_weights[abs(trained_weights)<hyper_p.node_TOL] = 0
        trained_biases[abs(trained_biases)<hyper_p.node_TOL] = 0
    trained_weights_dict = {"W"+str(l+1): trained_weights.flatten()}
    trained_biases_dict = {"b"+str(l+1): trained_biases.flatten()}
    df_trained_weights = pd.DataFrame(trained_weights_dict)
    df_trained_biases = pd.DataFrame(trained_biases_dict)
    df_trained_weights.to_csv(savefilepath + "_W" + str(l+1) + '.csv', index=False)
    df_trained_biases.to_csv(savefilepath + "_b" + str(l+1) + '.csv', index=False)
    
    #=== Save Output Weights and Biases ===# Note that these get replaced everytime
    trained_output_weights = sess.run("NN/W" + str(l+2) + ':0')
    trained_output_biases = sess.run("NN/b" + str(l+2) + ':0')
    if thresholding_flag == 1:
        trained_output_weights[abs(trained_output_weights)<hyper_p.node_TOL] = 0
        trained_output_biases[abs(trained_output_biases)<hyper_p.node_TOL] = 0
    trained_output_weights_dict = {"Woutput": trained_output_weights.flatten()}
    trained_output_biases_dict = {"boutput": trained_output_biases.flatten()}
    df_trained_output_weights = pd.DataFrame(trained_output_weights_dict)
    df_trained_output_biases = pd.DataFrame(trained_output_biases_dict)
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