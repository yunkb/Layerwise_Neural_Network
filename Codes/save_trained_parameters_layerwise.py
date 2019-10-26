#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:53:47 2019

@author: hwan
"""

import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Fully Connected                                 #
###############################################################################
def save_weights_and_biases_FC(sess, hyper_p, trainable_hidden_layer_index, savefilepath, thresholding_flag):
    #=== Save Newly Trained Weights and Biases ===#
    l = trainable_hidden_layer_index
    trained_weights = sess.run("NN/W" + str(l) + ':0')
    trained_biases = sess.run("NN/b" + str(l) + ':0')
    if thresholding_flag == 1:
        trained_weights[abs(trained_weights)<hyper_p.node_TOL] = 0
        trained_biases[abs(trained_biases)<hyper_p.node_TOL] = 0
    trained_weights_dict = {"W"+str(l): trained_weights.flatten()}
    trained_biases_dict = {"b"+str(l): trained_biases.flatten()}
    df_trained_weights = pd.DataFrame(trained_weights_dict)
    df_trained_biases = pd.DataFrame(trained_biases_dict)
    df_trained_weights.to_csv(savefilepath + "_W" + str(l) + '.csv', index=False)
    df_trained_biases.to_csv(savefilepath + "_b" + str(l) + '.csv', index=False)
    
    #=== Save Output Weights and Biases ===# Note that these get replaced everytime
    l += 1
    trained_output_weights = sess.run("NN/W" + str(l) + ':0')
    trained_output_biases = sess.run("NN/b" + str(l) + ':0')
    if thresholding_flag == 1:
        trained_output_weights[abs(trained_output_weights)<hyper_p.node_TOL] = 0
        trained_output_biases[abs(trained_output_biases)<hyper_p.node_TOL] = 0
    trained_output_weights_dict = {"Woutput": trained_output_weights.flatten()}
    trained_output_biases_dict = {"boutput": trained_output_biases.flatten()}
    df_trained_output_weights = pd.DataFrame(trained_output_weights_dict)
    df_trained_output_biases = pd.DataFrame(trained_output_biases_dict)
    df_trained_output_weights.to_csv(savefilepath + "_Woutput" + '.csv', index=False)
    df_trained_output_biases.to_csv(savefilepath + "_boutput" + '.csv', index=False)
    
###############################################################################
#                               Convolutional                                 #
###############################################################################
def save_weights_and_biases_CNN(sess, hyper_p, trainable_hidden_layer_index, savefilepath, thresholding_flag):
    #=== Save Input Feature Map Weights and Biases ===# Note that these get replaced everytime
    l = 1
    trained_weights = sess.run("NN/W" + str(l) + ':0')
    trained_biases = sess.run("NN/b" + str(l) + ':0')
    trained_weights_dict = {"Winput": trained_weights.flatten()}
    trained_biases_dict = {"binput": trained_biases.flatten()}
    df_trained_weights = pd.DataFrame(trained_weights_dict)
    df_trained_biases = pd.DataFrame(trained_biases_dict)
    df_trained_weights.to_csv(savefilepath + "_Winput" + '.csv', index=False)
    df_trained_biases.to_csv(savefilepath + "_binput" + '.csv', index=False)
    
    #=== Save Newly Trained Weights and Biases ===#
    l = trainable_hidden_layer_index
    trained_weights = sess.run("NN/W" + str(l) + ':0')
    trained_biases = sess.run("NN/b" + str(l) + ':0')
    if thresholding_flag == 1:
        trained_weights[abs(trained_weights)<hyper_p.node_TOL] = 0
        trained_biases[abs(trained_biases)<hyper_p.node_TOL] = 0
    trained_weights_dict = {"W"+str(l): trained_weights.flatten()}
    trained_biases_dict = {"b"+str(l): trained_biases.flatten()}
    df_trained_weights = pd.DataFrame(trained_weights_dict)
    df_trained_biases = pd.DataFrame(trained_biases_dict)
    df_trained_weights.to_csv(savefilepath + "_W" + str(l) + '.csv', index=False)
    df_trained_biases.to_csv(savefilepath + "_b" + str(l) + '.csv', index=False)
    
    #=== Save Downsampling Weights and Biases ===# Note that these get replaced everytime
    l = trainable_hidden_layer_index + 1
    trained_output_weights = sess.run("NN/W" + str(l) + ':0')
    trained_output_biases = sess.run("NN/b" + str(l) + ':0')
    trained_output_weights_dict = {"Wdownsample": trained_output_weights.flatten()}
    trained_output_biases_dict = {"bdownsample": trained_output_biases.flatten()}
    df_trained_output_weights = pd.DataFrame(trained_output_weights_dict)
    df_trained_output_biases = pd.DataFrame(trained_output_biases_dict)
    df_trained_output_weights.to_csv(savefilepath + "_Wdownsample" + '.csv', index=False)
    df_trained_output_biases.to_csv(savefilepath + "_bdownsample" + '.csv', index=False)
    
    #=== Save Output Weights and Biases ===# Note that these get replaced everytime
    l = trainable_hidden_layer_index + 2
    trained_output_weights = sess.run("NN/W" + str(l) + ':0')
    trained_output_biases = sess.run("NN/b" + str(l) + ':0')
    trained_output_weights_dict = {"Woutput": trained_output_weights.flatten()}
    trained_output_biases_dict = {"boutput": trained_output_biases.flatten()}
    df_trained_output_weights = pd.DataFrame(trained_output_weights_dict)
    df_trained_output_biases = pd.DataFrame(trained_output_biases_dict)
    df_trained_output_weights.to_csv(savefilepath + "_Woutput" + '.csv', index=False)
    df_trained_output_biases.to_csv(savefilepath + "_boutput" + '.csv', index=False)