#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_and_save_figures(hyper_p, run_options):
    print('================================')
    print('            Plotting            ')
    print('================================')
    
    first_trainable_hidden_layer_index = 2  
###############################################################################
#                                    Loss                                     #
###############################################################################       
    #=== Plot and Save Losses===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs-1, hyper_p.num_epochs-1, endpoint = True)
    for l in range(first_trainable_hidden_layer_index, hyper_p.max_hidden_layers):
        #=== Load Metrics and Plot ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(run_options.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_loss_array = array_metrics[2:,0]
        plt.plot(x_axis, storage_loss_array, label = 'hl' + str(l))
        
    #=== Figure Properties ===#   
    #plt.title('Training Loss on MNIST')
    plt.title(run_options.filename)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'loss' + '_all_layers_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

###############################################################################
#                                  Accuracy                                   #
###############################################################################
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs-1, hyper_p.num_epochs-1, endpoint = True)
    for l in range(first_trainable_hidden_layer_index, hyper_p.max_hidden_layers):
        #=== Load Metrics and Plot ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(run_options.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_accuracy_array = array_metrics[2:,2]
        plt.plot(x_axis, storage_accuracy_array, label = 'hl' + str(l))
        
    #=== Figure Properties ===#   
    #plt.title('Testing Accuracy on MNIST')
    plt.title(run_options.filename)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.axis([0,30,0.9,1])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'accuracy' + '_all_layers_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

###############################################################################
#                          Relative Number of Zeros                           #
###############################################################################        
    #=== Load Metrics and Plot ===#
    print('Loading relative number of zeros .csv file')
    try:
        df_rel_zeros = pd.read_csv(run_options.NN_savefile_name + "_relzeros" + '.csv')
        rel_zeros_array = df_rel_zeros.to_numpy()
        rel_zeros_array = rel_zeros_array.flatten()
    except:
        print('No relative number of zeros .csv file!')
    rel_zeros_array_exists = 'rel_zeros_array' in locals() or 'rel_zeros_array' in globals()
    
    if rel_zeros_array_exists:
        #=== Figure Properties ===# 
        fig_accuracy = plt.figure()
        x_axis = np.linspace(2, hyper_p.max_hidden_layers-1, hyper_p.max_hidden_layers-2, endpoint = True)
        plt.plot(x_axis, rel_zeros_array, label = 'relative # of 0s')
        plt.title(run_options.filename)
        plt.xlabel('Layer Number')
        plt.ylabel('Number of Zeros')
        plt.legend()
        
        #=== Saving Figure ===#
        figures_savefile_name = run_options.figures_savefile_directory + '/' + 'rel_num_zeros_' + run_options.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_accuracy)        
        
        
        
        
        
        
        
        
        