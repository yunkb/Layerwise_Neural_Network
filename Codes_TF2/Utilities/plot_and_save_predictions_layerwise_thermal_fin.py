#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dolfin as dl

import sys
sys.path.append('../..')

from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D
from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function
from Thermal_Fin_Heat_Simulator.Utilities.plot_2D import plot_2D
from Thermal_Fin_Heat_Simulator.Utilities.plot_3D import plot_3D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_and_save_predictions(hyperp, run_options, file_paths, fig_size):
###############################################################################
#                     Form Fenics Domain and Load Predictions                 #
###############################################################################
    #=== Form Fenics Domain ===#
    if run_options.fin_dimensions_2D == 1:
        V,_ = get_space_2D(40)
    if run_options.fin_dimensions_3D == 1:
        V, mesh = get_space_3D(40)
    
    solver = Fin(V) 
    
    #=== Load Observation Indices, Test and Predicted Parameters and State ===#
    df_obs_indices = pd.read_csv(file_paths.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy() 

    df_parameter_test = pd.read_csv(file_paths.savefile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    
    if run_options.forward_mapping == 1:
        df_state_pred = pd.read_csv(file_paths.savefile_name_state_pred + '.csv')
        state_pred = df_state_pred.to_numpy()
    
    if run_options.inverse_mapping == 1:
        df_parameter_pred = pd.read_csv(file_paths.savefile_name_parameter_pred + '.csv')
        parameter_pred = df_parameter_pred.to_numpy()
    
###############################################################################
#                     Form Fenics Domain and Load Predictions                 #
###############################################################################
    #=== Form Fenics Domain ===#
    if run_options.fin_dimensions_2D == 1:
        V,_ = get_space_2D(40)
    if run_options.fin_dimensions_3D == 1:
        V, mesh = get_space_3D(40)
    
    solver = Fin(V) 
    
    #=== Load Observation Indices, Test and Predicted Parameters and State ===#
    df_obs_indices = pd.read_csv(file_paths.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy() 

    df_parameter_test = pd.read_csv(file_paths.savefile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    
    if run_options.forward_mapping == 1:
        df_state_pred = pd.read_csv(file_paths.savefile_name_state_pred + '.csv')
        state_pred = df_state_pred.to_numpy()
    
    if run_options.inverse_mapping == 1:
        df_parameter_pred = pd.read_csv(file_paths.savefile_name_parameter_pred + '.csv')
        parameter_pred = df_parameter_pred.to_numpy()
    
###############################################################################
#                             Plotting Predictions                            #
###############################################################################
    #=== Converting Test Parameter Into Dolfin Object and Computed State Observation ===#       
    if run_options.data_thermal_fin_nine == 1:
        parameter_test_dl = solver.nine_param_to_function(parameter_test)
        if run_options.fin_dimensions_3D == 1: # Interpolation messes up sometimes and makes some values equal 0
            parameter_values = parameter_test_dl.vector().get_local()  
            zero_indices = np.where(parameter_values == 0)[0]
            for ind in zero_indices:
                parameter_values[ind] = parameter_values[ind-1]
            parameter_test_dl = convert_array_to_dolfin_function(V, parameter_values)
    if run_options.data_thermal_fin_vary == 1:
        parameter_test_dl = convert_array_to_dolfin_function(V,parameter_test)
    
    state_test_dl, _ = solver.forward(parameter_test_dl) # generate true state for comparison
    state_test = state_test_dl.vector().get_local()    
    if hyperp.data_type == 'bnd':
        state_test = state_test[obs_indices].flatten()
    
    #=== Plotting Test Parameter and Test State ===#  
    if run_options.fin_dimensions_2D == 1:
        p_test_fig, ax = plot_2D(parameter_test_dl, 'True Parameter', fig_size)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(p_test_fig, cax = cax)  
    if run_options.fin_dimensions_3D == 1:
        p_test_fig, ax = plot_3D(parameter_test_dl, 'True Parameter', angle_1 = 90, angle_2 = 270, fig_size = fig_size)
        caxis = inset_axes(ax, width="5%", height="60%", loc='right')
        plt.colorbar(p_test_fig, cax=caxis, ticks=[0.5,1.0,1.5,2.0,2.5,3.0]) 
    plt.savefig(file_paths.figures_savefile_name_parameter_test, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    print('Figure saved to ' + file_paths.figures_savefile_name_parameter_test)   
    plt.show()
    
    if hyperp.data_type == 'full': # No state prediction for bnd only data
        if run_options.fin_dimensions_2D == 1:
            s_test_fig, ax = plot_2D(state_test_dl, 'True State', fig_size)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(s_test_fig, cax = cax)
        if run_options.fin_dimensions_3D == 1:
            s_test_fig, ax = plot_3D(state_test_dl, 'True State', angle_1 = 90, angle_2 = 270, fig_size = fig_size)
            caxis = inset_axes(ax, width="5%", height="60%", loc='right')
            plt.colorbar(s_test_fig, cax=caxis, ticks=[0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4]) # piecewise constant
            #plt.colorbar(s_test_fig, cax=caxis, ticks=[0.0,0.2,0.4,0.6,0.8,1.0]) # spatially varying
        plt.savefig(file_paths.figures_savefile_name_state_test, dpi=300, bbox_inches = 'tight', pad_inches = 0)
        print('Figure saved to ' + file_paths.figures_savefile_name_state_test) 
        plt.show()
    
    #=== Converting Predicted Parameter into Dolfin Object ===# 
    if run_options.inverse_mapping == 1:
        if run_options.data_thermal_fin_nine == 1:
            parameter_pred_dl = solver.nine_param_to_function(parameter_pred)
            if run_options.fin_dimensions_3D == 1: # Interpolation messes up sometimes and makes some values equal 0
                parameter_values = parameter_pred_dl.vector().get_local()  
                zero_indices = np.where(parameter_values == 0)[0]
                for ind in zero_indices:
                    parameter_values[ind] = parameter_values[ind-1]
                parameter_pred_dl = convert_array_to_dolfin_function(V, parameter_values)
        if run_options.data_thermal_fin_vary == 1:
            parameter_pred_dl = convert_array_to_dolfin_function(V,parameter_pred)   
    
    #=== Plotting Predicted Parameter and State ===#
    if run_options.inverse_mapping == 1:
        if run_options.fin_dimensions_2D == 1:
            p_pred_fig, ax = plot_2D(parameter_pred_dl, 'Inverse Estimation of True Parameter', fig_size)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(p_test_fig, cax = cax)
        if run_options.fin_dimensions_3D == 1:
            p_pred_fig, ax = plot_3D(parameter_pred_dl, 'Inverse Estimation of True Parameter', angle_1 = 90, angle_2 = 270, fig_size = fig_size)
            caxis = inset_axes(ax, width="5%", height="60%", loc='right')
            plt.colorbar(p_test_fig, cax=caxis, ticks=[0.5,1.0,1.5,2.0,2.5,3.0]) 
        plt.savefig(file_paths.figures_savefile_name_parameter_pred, dpi=300, bbox_inches = 'tight', pad_inches = 0)
        print('Figure saved to ' + file_paths.figures_savefile_name_parameter_pred) 
        plt.show()
        parameter_pred_error = np.linalg.norm(parameter_pred - parameter_test,2)/np.linalg.norm(parameter_test,2)
        print('Parameter prediction relative error: %.7f' %parameter_pred_error)
    
    if run_options.forward_mapping == 1:
        if hyperp.data_type == 'full': # No visualization of state prediction if the truncation layer only consists of the boundary observations
            state_pred_dl = convert_array_to_dolfin_function(V, state_pred)
            if run_options.fin_dimensions_2D == 1:
                s_pred_fig, ax = plot_2D(state_pred_dl, 'Forward Estimation of True State', fig_size)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(s_test_fig, cax = cax)
            if run_options.fin_dimensions_3D == 1:
                s_pred_fig, ax = plot_3D(state_pred_dl, 'Forward Estimation of True State', angle_1 = 90, angle_2 = 270, fig_size = fig_size)
                caxis = inset_axes(ax, width="5%", height="60%", loc='right')
                plt.colorbar(s_test_fig, cax=caxis, ticks=[0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4]) # piecewise constant
                #plt.colorbar(s_test_fig, cax=caxis, ticks=[0.0,0.2,0.4,0.6,0.8,1.0]) # spatially varying 
            plt.savefig(file_paths.figures_savefile_name_state_pred, dpi=300, bbox_inches = 'tight', pad_inches = 0)
            print('Figure saved to ' + file_paths.figures_savefile_name_state_pred) 
            plt.show()
        state_pred_error = np.linalg.norm(state_pred - state_test,2)/np.linalg.norm(state_test,2)
        print('State observation prediction relative error: %.7f' %state_pred_error)
        
###############################################################################
#                               Plotting Metrics                              #
###############################################################################     
    plt.ioff() # Turn interactive plotting off
    first_trainable_hidden_layer_index = 2  
    marker_list = ['+', '*', 'x', 'D', 'o', '.', 'h']
    
############
#   Loss   #
############    
    #=== Plot and Save Losses===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs-1, hyperp.num_epochs-1, endpoint = True)
    for l in range(first_trainable_hidden_layer_index, hyperp.max_hidden_layers):
        #=== Load Metrics and Plot ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_loss_array = array_metrics[2:,0]
        plt.plot(x_axis, np.log(storage_loss_array), label = 'hl' + str(l), marker = marker_list[l-2])
        
    #=== Figure Properties ===#   
    plt.title('Training Log-Loss')
    #plt.title(file_paths.filename)
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_all_layers_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

################
#   Accuracy   #
################
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs-1, hyperp.num_epochs-1, endpoint = True)
    for l in range(first_trainable_hidden_layer_index, hyperp.max_hidden_layers):
        #=== Load Metrics and Plot ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_accuracy_array = array_metrics[2:,1]
        plt.plot(x_axis, storage_accuracy_array, label = 'hl' + str(l), marker = marker_list[l-2])
        
    #=== Figure Properties ===#   
    plt.title('Testing Relative Errors')
    #plt.title(file_paths.filename)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.axis([0,30,0.9,1])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'accuracy' + '_all_layers_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

################################
#   Relative Number of Zeros   #
################################     
    #=== Load Metrics and Plot ===#
    print('Loading relative number of zeros .csv file')
    try:
        df_rel_zeros = pd.read_csv(file_paths.NN_savefile_name + "_relzeros" + '.csv')
        rel_zeros_array = df_rel_zeros.to_numpy()
        rel_zeros_array = rel_zeros_array.flatten()
    except:
        print('No relative number of zeros .csv file!')
    rel_zeros_array_exists = 'rel_zeros_array' in locals() or 'rel_zeros_array' in globals()
    
    if rel_zeros_array_exists:
        #=== Figure Properties ===# 
        fig_accuracy = plt.figure()
        x_axis = np.linspace(2, hyperp.max_hidden_layers-1, hyperp.max_hidden_layers-2, endpoint = True)
        plt.plot(x_axis, rel_zeros_array, label = 'relative # of 0s')
        plt.title('Relative Number of Zeros')
        plt.xlabel('Layer Number')
        plt.ylabel('Relative Number of Zeros')
        plt.legend()
        
        #=== Saving Figure ===#
        figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'rel_num_zeros_' + file_paths.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_accuracy)        
        
        
        
        
        
        
        
        
        