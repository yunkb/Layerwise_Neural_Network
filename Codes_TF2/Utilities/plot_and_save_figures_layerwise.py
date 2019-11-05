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

###############################################################################
#                           Investigating Results                             #
###############################################################################
def plot_and_save_figures(hyper_p, run_options):
    
    if run_options.NN_type == 'FC':
        first_trainable_hidden_layer_index = 1
    if run_options.NN_type == 'CNN':
        first_trainable_hidden_layer_index = 2    
    
    print('================================')
    print('            Plotting            ')
    print('================================')
    #=== Plotting ===#    
    print(run_options.filename)
    for l in range(first_trainable_hidden_layer_index, hyper_p.max_hidden_layers):
        #=== Load Metrics ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(run_options.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_loss_array = array_metrics[:,0]
        storage_accuracy_array = array_metrics[:,1]  
        
        #=== Plot and Save Losses===#
        fig_loss = plt.figure()
        x_axis = np.linspace(0, hyper_p.num_epochs+1, hyper_p.num_epochs+1, endpoint = True)
        plt.plot(x_axis, storage_loss_array, label = 'hl' + str(l))
        plt.title('Loss for: ' + run_options.filename)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        figures_savefile_name = run_options.figures_savefile_directory + '/' + 'loss' + '_hl' + str(l) + '_' + run_options.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_loss)
        
        #=== Plot and Save Accuracies===#
        fig_accuracy = plt.figure()
        x_axis = np.linspace(0, hyper_p.num_epochs+1, hyper_p.num_epochs+1, endpoint = True)
        plt.plot(x_axis, storage_accuracy_array, label = 'hl' + str(l))
        plt.title('Accuracy for: ' + run_options.filename)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        figures_savefile_name = run_options.figures_savefile_directory + '/' + 'accuracy' + '_hl' + str(l) + '_' + run_options.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_accuracy)
        
    #=== Plot and Save Relative Number of Zeros ===#
    print('Loading relative number of zeros .csv file')
    try:
        df_rel_zeros = pd.read_csv(run_options.NN_savefile_name + "_relzeros" + '.csv')
        rel_zeros_array = df_rel_zeros.to_numpy()
        rel_zeros_array = rel_zeros_array.flatten()
    except:
        print('No relative number of zeros .csv file!')
    rel_zeros_array_exists = 'rel_zeros_array' in locals() or 'rel_zeros_array' in globals()
    if rel_zeros_array_exists:
        fig_accuracy = plt.figure()
        x_axis = np.linspace(2, hyper_p.max_hidden_layers-1, hyper_p.max_hidden_layers-2, endpoint = True)
        plt.plot(x_axis, rel_zeros_array, label = 'relative # of 0s')
        plt.title('Rel # of 0s: ' + run_options.filename)
        plt.xlabel('Layer Number')
        plt.ylabel('Number of Zeros')
        plt.legend()
        figures_savefile_name = run_options.figures_savefile_directory + '/' + 'rel_num_zeros_' + run_options.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_accuracy)
        
###############################################################################
#                              Results for Paper                              #
###############################################################################
def plot_and_save_figures_paper_results(hyper_p, run_options):
    
    if run_options.NN_type == 'FC':
        first_trainable_hidden_layer_index = 1
    if run_options.NN_type == 'CNN':
        first_trainable_hidden_layer_index = 2    
    
    print('================================')
    print('            Plotting            ')
    print('================================')
    #=== Plotting ===#    
    print(run_options.filename)        
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
    plt.title('Training Loss on MNIST')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.axis([0,30,0,1.2])
    plt.legend()
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'loss' + '_all_layers_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)
    
    #=== Plot and Save Accuracies===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs-1, hyper_p.num_epochs-1, endpoint = True)
    for l in range(first_trainable_hidden_layer_index, hyper_p.max_hidden_layers):
        #=== Load Metrics and Plot ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(run_options.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_accuracy_array = array_metrics[2:,1]
        plt.plot(x_axis, storage_accuracy_array, label = 'hl' + str(l))
    plt.title('Testing Accuracy on MNIST')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.axis([0,30,0.9,1])
    plt.legend()
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'accuracy' + '_all_layers_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)
        
    #=== Plot and Save Relative Number of Zeros ===#
    print('Loading relative number of zeros .csv file')
    try:
        df_rel_zeros = pd.read_csv(run_options.NN_savefile_name + "_relzeros" + '.csv')
        rel_zeros_array = df_rel_zeros.to_numpy()
        rel_zeros_array = rel_zeros_array.flatten()
    except:
        print('No relative number of zeros .csv file!')
    rel_zeros_array_exists = 'rel_zeros_array' in locals() or 'rel_zeros_array' in globals()
    if rel_zeros_array_exists:
        fig_accuracy = plt.figure()
        x_axis = np.linspace(2, hyper_p.max_hidden_layers-1, hyper_p.max_hidden_layers-2, endpoint = True)
        plt.plot(x_axis, rel_zeros_array, label = 'relative # of 0s')
        plt.title('Rel # of 0s: ' + run_options.filename)
        plt.xlabel('Layer Number')
        plt.ylabel('Number of Zeros')
        plt.legend()
        figures_savefile_name = run_options.figures_savefile_directory + '/' + 'rel_num_zeros_' + run_options.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_accuracy)        
        
        
        
        
        
        
        
        
        