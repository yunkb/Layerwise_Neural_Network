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
    trained_weights = {"W"+str(l+1): [sess.run("NN_layerwise/W" + str(l+1) + ':0')]}
    trained_biases = {"b"+str(l+1): [sess.run("NN_layerwise/b" + str(l+1) + ':0')]}
    df_trained_weights = pd.DataFrame(trained_weights)
    df_trained_biases = pd.DataFrame(trained_biases)
    df_trained_weights.to_csv(savefilepath + "_W" + str(l+1) + '.csv', index=False)
    df_trained_biases.to_csv(savefilepath + "_b" + str(l+1) + '.csv', index=False)
    
    #=== Save Output Weights and Biases ===# Note that these get replaced everytime
    trained_output_weights = {"Woutput": [sess.run("NN_layerwise/W" + str(l+2) + ':0')]}
    trained_output_biases = {"boutput"+str(l+2): [sess.run("NN_layerwise/b" + str(l+2) + ':0')]}
    df_trained_output_weights = pd.DataFrame(trained_output_weights)
    df_trained_output_biases = pd.DataFrame(trained_output_biases)
    df_trained_output_weights.to_csv(savefilepath + "_Woutput" + '.csv', index=False)
    df_trained_output_biases.to_csv(savefilepath + "_boutput" + '.csv', index=False)
    
    pdb.set_trace()
    
    