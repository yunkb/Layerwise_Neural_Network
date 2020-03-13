#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:46:01 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from Utilities.get_thermal_fin_data import load_thermal_fin_test_data
from Utilities.NN_FC_layerwise import FCLayerwise

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def predict_and_save(hyperp, run_options, file_paths):    
    #=== Load Testing Data ===# 
    obs_indices, parameter_test, state_obs_test, data_input_shape_temp, parameter_dimension\
    = load_thermal_fin_test_data(file_paths, run_options.num_data_test, run_options.parameter_dimensions) 
    output_dimensions_temp = len(obs_indices)

    #=== Shuffling Data and Forming Batches ===#
    parameter_and_state_obs_test = tf.data.Dataset.from_tensor_slices((parameter_test, state_obs_test)).shuffle(8192, seed=run_options.random_seed).batch(hyperp.batch_size)

    #=== Forward or Inverse Mapping Input and Output Dimensions ===#
    if run_options.forward_mapping == 1:
        data_input_shape = data_input_shape_temp
        output_dimensions = output_dimensions_temp
    if run_options.inverse_mapping == 1:
        data_input_shape = np.array([output_dimensions_temp, 1])
        output_dimensions = data_input_shape_temp[0]

    ####################################
    #   Import Trained Neural Network  #
    ####################################        
    #=== Neural Network ===#
    kernel_regularizer = tf.keras.regularizers.l1(hyperp.regularization)
    bias_regularizer = tf.keras.regularizers.l1(hyperp.regularization)
    NN = FCLayerwise(hyperp, run_options, data_input_shape, output_dimensions,
                     kernel_regularizer, bias_regularizer)    
    NN.load_weights(file_paths.NN_savefile_name)     
    
    #######################
    #   Form Predictions  #
    #######################      
    #=== From Parameter Instance ===#
    df_parameter_test = pd.read_csv(file_paths.loadfile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_state_test = pd.read_csv(file_paths.loadfile_name_state_test + '.csv')
    state_test = df_state_test.to_numpy()
    
    if run_options.forward_mapping == 1:
        state_pred = NN(parameter_test.T)
        state_pred = state_pred.numpy().flatten()
        
    if run_options.inverse_mapping == 1:
        if hyperp.data_type == 'bnd':
            state_test_bnd = state_test[obs_indices].flatten()
            state_test_bnd = state_test_bnd.reshape(state_test_bnd.shape[0], 1)
            parameter_pred = NN(state_test_bnd.T) 
        else:
            parameter_pred = NN(state_test.T) 
        parameter_pred = parameter_pred.numpy().flatten()
        
    parameter_test = parameter_test.flatten()
    state_test = state_test.flatten()
    
# =============================================================================
#     #=== From Test Batch ===#
#     parameter_and_state_obs_test_draw = parameter_and_state_obs_test.take(1)
#     for batch_num, (parameter_test, state_obs_test) in parameter_and_state_obs_test_draw.enumerate():
#         state_pred_batch = NN(parameter_test)
#           
#     parameter_test = parameter_test[4,:].numpy()
#     state_test = state_obs_test[4,:].numpy()
#     state_pred = state_pred_batch[4,:].numpy()
# =============================================================================
    
    #=== Generating Boundary Data from Full Data ===#
    #state_test = state_test[obs_indices].flatten()
    
    #####################################
    #   Save Test Case and Predictions  #
    #####################################  
    df_parameter_test = pd.DataFrame({'parameter_test': parameter_test})
    df_parameter_test.to_csv(file_paths.savefile_name_parameter_test + '.csv', index=False)  
    df_state_test = pd.DataFrame({'state_test': state_test})
    df_state_test.to_csv(file_paths.savefile_name_state_test + '.csv', index=False)  
    if run_options.inverse_mapping == 1:
        df_parameter_pred = pd.DataFrame({'parameter_pred': parameter_pred})
        df_parameter_pred.to_csv(file_paths.savefile_name_parameter_pred + '.csv', index=False)
    if run_options.forward_mapping == 1:
        df_state_pred = pd.DataFrame({'state_pred': state_pred})
        df_state_pred.to_csv(file_paths.savefile_name_state_pred + '.csv', index=False)  

    print('\nPredictions Saved to ' + file_paths.NN_savefile_directory)
        
    