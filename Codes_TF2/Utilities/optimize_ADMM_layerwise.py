#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:35:17 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from Utilities.ADMM_methods import initialize_z_and_lagrange_multiplier, update_ADMM_penalty_terms, update_z_and_lagrange_multiplier

import shutil # for deleting directories
import os
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_ADMM(hyper_p, run_options, file_paths, NN, data_loss, accuracy, data_and_labels_train, data_and_labels_test, data_and_labels_val, label_dimensions, num_batches_train):
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()
    reset_optimizer = tf.group([v.initializer for v in optimizer.variables()])

    #=== Define Metrics and Initialize Metric Storage Arrays ===#
    data_loss_train_batch_average = tf.keras.metrics.Mean()
    data_loss_val_batch_average = tf.keras.metrics.Mean()
    data_loss_test_batch_average = tf.keras.metrics.Mean()
    loss_test_batch_average = tf.keras.metrics.Mean()
    loss_train_batch_average = tf.keras.metrics.Mean()
    loss_val_batch_average = tf.keras.metrics.Mean()
    accuracy_train_batch_average = tf.keras.metrics.Mean()
    accuracy_val_batch_average = tf.keras.metrics.Mean()
    accuracy_test_batch_average = tf.keras.metrics.Mean()
    storage_loss_array = np.array([])
    storage_accuracy_array = np.array([])
    
    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists(file_paths.tensorboard_directory): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree(file_paths.tensorboard_directory)  
    summary_writer = tf.summary.create_file_writer(file_paths.tensorboard_directory)

###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    data_loss_validation = 1e5
    trainable_hidden_layer_index = 2
    relative_number_zeros = 0
    storage_data_loss_array = []
    storage_loss_array = []
    storage_accuracy_array = []
    storage_relative_number_zeros_array = []
    
    #####################################
    #   Training Current Architecture   #
    #####################################
    while data_loss_validation > hyper_p.error_TOL and trainable_hidden_layer_index < hyper_p.max_hidden_layers:    
        #=== Initial Loss and Accuracy ===#
        for batch_num, (data_train, labels_train) in data_and_labels_train.enumerate():
            output = NN(data_train)
            data_loss_train_batch = data_loss(output, labels_train, label_dimensions)
            loss_train_batch = data_loss_train_batch # ADMM penalty equals 0
            data_loss_train_batch_average(data_loss_train_batch)
            loss_train_batch_average(loss_train_batch) 
            accuracy_train_batch_average(accuracy(output, labels_train))
        for data_val, labels_val in data_and_labels_val:
            output_val = NN(data_val)
            data_loss_val_batch = data_loss(output, labels_train, label_dimensions)
            loss_val_batch = data_loss_val_batch # ADMM penalty equals 0
            data_loss_train_batch_average(data_loss_val_batch)
            loss_val_batch_average(loss_val_batch)
            accuracy_val_batch_average(accuracy(output_val, labels_val))
        for data_test, labels_test in data_and_labels_test:
            output_test = NN(data_test)
            data_loss_test_batch = data_loss(output_test, labels_test, label_dimensions)
            loss_test_batch = data_loss_test_batch # ADMM penalty equals 0
            loss_test_batch_average(loss_test_batch)
            accuracy_test_batch_average(accuracy(output_test, labels_test))
        storage_data_loss_array = np.append(storage_data_loss_array, data_loss_train_batch_average.result())
        storage_loss_array = np.append(storage_loss_array, loss_train_batch_average.result())
        storage_accuracy_array = np.append(storage_accuracy_array, accuracy_test_batch_average.result())
        print('Initial Losses:')
        print('Training Set: Loss: %.3e, Accuracy: %.3f' %(loss_train_batch_average.result(), accuracy_train_batch_average.result()))
        print('Validation Set: Loss: %.3e, Accuracy: %.3f\n' %(loss_val_batch_average.result(), accuracy_val_batch_average.result()))
        print('Test Set: Loss: %.3e, Accuracy: %.3f\n' %(loss_test_batch_average.result(), accuracy_test_batch_average.result()))
        
        #==== Beginning Training ===#
        print('Beginning Training')
        for epoch in range(hyper_p.num_epochs):
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(file_paths.filename)
            print('Trainable Hidden Layer Index: %d' %(trainable_hidden_layer_index))
            print('GPU: ' + run_options.which_gpu + '\n')
            print('Optimizing %d batches of size %d:' %(num_batches_train, hyper_p.batch_size))
            start_time_epoch = time.time()
            for batch_num, (data_train, labels_train) in data_and_labels_train.enumerate():
                with tf.GradientTape() as tape:
                    start_time_batch = time.time()
                    output = NN(data_train)
                    #=== Display Model Summary ===#
                    if batch_num == 0 and epoch == 0:
                        NN.summary()
                        z, lagrange = initialize_z_and_lagrange_multiplier(NN.get_weights()) 
                    ADMM_penalty = update_ADMM_penalty_terms(hyper_p.penalty, NN.weights, z, lagrange)
                    data_loss_train_batch = data_loss(output, labels_train, label_dimensions) 
                    loss_train_batch = data_loss_train_batch + ADMM_penalty
                    gradients = tape.gradient(loss_train_batch, NN.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                    elapsed_time_batch = time.time() - start_time_batch
                    if batch_num  == 0:
                        print('Time per Batch: %.2f' %(elapsed_time_batch))
                    #=== Update ADMM Objects ===#
                    if batch_num != 0 and batch_num % 10 == 0 and epoch >= 5: # Warm start and updating before w is fully minimized
                        z, lagrange = update_z_and_lagrange_multiplier(NN.get_weights(), hyper_p.regularization, hyper_p.penalty, z, lagrange)
                data_loss_train_batch_average(data_loss_train_batch)
                loss_train_batch_average(loss_train_batch) 
                accuracy_train_batch_average(accuracy(output, labels_train))
                        
            #=== Computing Validation Metrics ===#
            for data_val, labels_val in data_and_labels_val:
                output_val = NN(data_val)
                data_loss_val_batch = data_loss(output_val, labels_val, label_dimensions)
                loss_val_batch = data_loss_val_batch + ADMM_penalty
                data_loss_val_batch_average(data_loss_val_batch)
                loss_val_batch_average(loss_val_batch)
                accuracy_val_batch_average(accuracy(output_val, labels_val))
                
            #=== Computing Testing Metrics ===#
            for data_test, labels_test in data_and_labels_test:
                output_test = NN(data_test)
                data_loss_test_batch = data_loss(output_test, labels_test, label_dimensions)
                loss_test_batch += data_loss_test_batch + ADMM_penalty
                data_loss_test_batch_average(data_loss_test_batch)
                loss_test_batch_average(loss_test_batch)
                accuracy_test_batch_average(accuracy(output_test, labels_test))
            
            #=== Track Training Metrics, Weights and Gradients ===#
            with summary_writer.as_default():
                tf.summary.scalar('data_loss_training', data_loss_train_batch_average.result(), step=epoch)
                tf.summary.scalar('loss_training', loss_train_batch_average.result(), step=epoch)
                tf.summary.scalar('accuracy_training', accuracy_train_batch_average.result(), step=epoch)
                tf.summary.scalar('data_loss_validation', data_loss_val_batch_average.result(), step=epoch)
                tf.summary.scalar('loss_validation', loss_val_batch_average.result(), step=epoch)
                tf.summary.scalar('accuracy_validation', accuracy_val_batch_average.result(), step=epoch)
                tf.summary.scalar('data_loss_testing', data_loss_test_batch_average.result(), step=epoch)
                tf.summary.scalar('loss_test', loss_test_batch_average.result(), step=epoch)
                tf.summary.scalar('accuracy_test', accuracy_test_batch_average.result(), step=epoch)
                storage_data_loss_array = np.append(storage_data_loss_array, data_loss_train_batch_average.result())
                storage_loss_array = np.append(storage_loss_array, loss_train_batch_average.result())
                storage_accuracy_array = np.append(storage_accuracy_array, accuracy_test_batch_average.result())
                for w in NN.weights:
                    tf.summary.histogram(w.name, w, step=epoch)
                l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
                for gradient, variable in zip(gradients, NN.trainable_variables):
                    tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)
                
            #=== Display Epoch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
            print('Training Set: Data Loss: %.3e, Loss: %.3e, Accuracy: %.3f' %(data_loss_train_batch_average.result(), loss_train_batch_average.result(), accuracy_train_batch_average.result()))
            print('Validation Set: Data Loss: %.3e, Loss: %.3e, Accuracy: %.3f' %(data_loss_val_batch_average.result(), loss_val_batch_average.result(), accuracy_val_batch_average.result()))
            print('Test Set: Data_Loss: %.3e, Loss: %.3e, Accuracy: %.3f\n' %(data_loss_test_batch_average.result(), loss_test_batch_average.result(), accuracy_test_batch_average.result()))
            print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            start_time_epoch = time.time() 
            
            #=== Reset Metrics ===#
            data_loss_validation = data_loss_val_batch_average.result()
            data_loss_train_batch_average.reset_states()
            loss_train_batch_average.reset_states()
            accuracy_train_batch_average.reset_states()
            data_loss_val_batch_average.reset_states()
            loss_val_batch_average.reset_states()
            accuracy_val_batch_average.reset_states()
            data_loss_test_batch_average.reset_states()
            loss_test_batch_average.reset_states()
            accuracy_test_batch_average.reset_states()
        
        ########################################################
        #   Updating Architecture and Saving Current Metrics   #
        ########################################################  
        print('================================')
        print('     Extending Architecture     ')
        print('================================')          
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['data_loss'] = storage_data_loss_array
        metrics_dict['loss'] = storage_loss_array
        metrics_dict['accuracy'] = storage_accuracy_array
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv', index=False)
        
        #=== Sparsify Weights of Trained Layer ===#
        relative_number_zeros = NN.sparsify_weights_and_get_relative_number_of_zeros(hyper_p.node_TOL)
        print('Relative Number of Zeros for Last Layer: %d\n' %(relative_number_zeros))
        storage_relative_number_zeros_array = np.append(storage_relative_number_zeros_array, relative_number_zeros)
          
        #=== Saving Relative Number of Zero Elements ===#
        relative_number_zeros_dict = {}
        relative_number_zeros_dict['rel_zeros'] = storage_relative_number_zeros_array
        df_relative_number_zeros = pd.DataFrame(relative_number_zeros_dict)
        df_relative_number_zeros.to_csv(file_paths.NN_savefile_name + "_relzeros" + '.csv', index=False)  
        
        #=== Add Layer ===#
        trainable_hidden_layer_index += 1
        NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)
        
        #=== Preparing for Next Training Cycle ===#
        storage_data_loss_array = []
        storage_loss_array = []
        storage_accuracy_array = []
        reset_optimizer        
    
    ########################
    #   Save Final Model   #
    ########################  
    #=== Saving Trained Model ===#          
    NN.save_weights(file_paths.NN_savefile_name)
    print('Final Model Saved') 
        

    
