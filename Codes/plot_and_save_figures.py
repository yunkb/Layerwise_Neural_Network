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
    
    if run_options.NN_type == 'FC':
        first_trainable_hidden_layer_index = 1
    if run_options.NN_type == 'CNN':
        first_trainable_hidden_layer_index = 2    
    
    #=== Plotting ===#    
    for l in range(first_trainable_hidden_layer_index, hyper_p.max_hidden_layers):
        #=== Load Metrics ===#
        print('Loading Metrics for Hidden Layer %d' %(l))
        df_metrics = pd.read_csv(run_options.NN_savefile_name + "_metrics_hl" + str(l) + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_loss_array = array_metrics[:,0]
        storage_accuracy_array = array_metrics[:,1]   
        
        #=== Plot and Save Losses===#
        fig_loss = plt.figure()
        x_axis = np.linspace(1, hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
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
        x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
        plt.plot(x_axis, storage_accuracy_array, label = 'hl' + str(l))
        plt.title('Accuracy for: ' + run_options.filename)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        figures_savefile_name = run_options.figures_savefile_directory + '/' + 'accuracy' + '_hl' + str(l) + '_' + run_options.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_accuracy)