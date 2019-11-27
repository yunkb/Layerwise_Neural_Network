#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:35:17 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np
import pandas as pd

import shutil # for deleting directories
import os
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, run_options, file_paths, NN, data_loss, accuracy, data_and_labels_train, data_and_labels_val, data_and_labels_test, label_dimensions, num_batches_train):
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()
    reset_optimizer = tf.group([v.initializer for v in optimizer.variables()])
    
    #=== Metrice ===#
    mean_loss_train = tf.keras.metrics.Mean()
    mean_loss_val = tf.keras.metrics.Mean()
    mean_loss_test = tf.keras.metrics.Mean()
    mean_accuracy_train = tf.keras.metrics.Mean()
    mean_accuracy_val = tf.keras.metrics.Mean()
    mean_accuracy_test = tf.keras.metrics.Mean()
    
    #=== Initialize Metric Storage Arrays ===#
    storage_array_loss = np.array([])
    storage_array_accuracy = np.array([])
    storage_array_relative_number_zeros = np.array([])
    
    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists('../Tensorboard/' + file_paths.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + file_paths.filename)  
    summary_writer = tf.summary.create_file_writer('../Tensorboard/' + file_paths.filename)

###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    loss_validation = 1e5
    trainable_hidden_layer_index = 2
    relative_number_zeros = 0
    
    #####################################
    #   Training Current Architecture   #
    #####################################
    while loss_validation > hyperp.error_TOL and trainable_hidden_layer_index < hyperp.max_hidden_layers:    
        #=== Initial Loss and Accuracy ===#
        for batch_num, (batch_data_train, batch_labels_train) in data_and_labels_train.enumerate():
            batch_pred_train = NN(batch_data_train)
            batch_loss_train = tf.reduce_mean(data_loss(batch_pred_train, batch_labels_train, label_dimensions))
            batch_loss_train += sum(NN.losses)
            mean_loss_train(batch_loss_train) 
            mean_accuracy_train(tf.reduce_mean(accuracy(batch_pred_train, batch_labels_train)))
        for batch_data_val, batch_labels_val in data_and_labels_val:
            batch_pred_val = NN(batch_data_val)
            batch_loss_val = tf.reduce_mean(data_loss(batch_pred_val, batch_labels_val, label_dimensions))
            batch_loss_val += sum(NN.losses)
            mean_loss_val(batch_loss_val)
            mean_accuracy_val(tf.reduce_mean(accuracy(batch_pred_val, batch_labels_val)))
        for batch_data_test, batch_labels_test in data_and_labels_test:
            batch_pred_test = NN(batch_data_test)
            batch_loss_test = tf.reduce_mean(data_loss(batch_pred_test, batch_labels_test, label_dimensions))
            batch_loss_test += sum(NN.losses)
            mean_loss_test(batch_loss_test)
            mean_accuracy_test(tf.reduce_mean(accuracy(batch_pred_test, batch_labels_test)))
        storage_array_loss = np.append(storage_array_loss, mean_loss_train.result())
        storage_array_accuracy = np.append(storage_array_accuracy, mean_accuracy_test.result())
        print('Initial Losses:')
        print('Training Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_train.result(), mean_accuracy_train.result()))
        print('Validation Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_val.result(), mean_accuracy_val.result()))
        print('Test Set: Loss: %.3e, Accuracy: %.3f\n' %(mean_loss_test.result(), mean_accuracy_test.result()))
        
        #=== Begin Training ===#
        print('Beginning Training')
        for epoch in range(hyperp.num_epochs):
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(file_paths.filename)
            print('Trainable Hidden Layer Index: %d' %(trainable_hidden_layer_index))
            print('GPU: ' + run_options.which_gpu + '\n')
            print('Optimizing %d batches of size %d:' %(num_batches_train, hyperp.batch_size))
            start_time_epoch = time.time()
            for batch_num, (batch_data_train, batch_labels_train) in data_and_labels_train.enumerate():
                with tf.GradientTape() as tape:
                    start_time_batch = time.time()
                    batch_pred_train = NN(batch_data_train)
                    #=== Display Model Summary ===#
                    if batch_num == 0 and epoch == 0:
                        NN.summary()
                    batch_loss_train = tf.reduce_mean(data_loss(batch_pred_train, batch_labels_train, label_dimensions))
                    batch_loss_train += sum(NN.losses)
                gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
                optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
                mean_loss_train(batch_loss_train) 
                mean_accuracy_train(tf.reduce_mean(accuracy(batch_pred_train, batch_labels_train)))
                                        
            #=== Computing Validation Metrics ===#
            for batch_data_val, batch_labels_val in data_and_labels_val:
                batch_pred_val = NN(batch_data_val)
                batch_loss_val = tf.reduce_mean(data_loss(batch_pred_val, batch_labels_val, label_dimensions))
                batch_loss_val += sum(NN.losses)
                mean_loss_val(batch_loss_val)
                mean_accuracy_val(tf.reduce_mean(accuracy(batch_pred_val, batch_labels_val)))
            
            #=== Computing Testing Metrics ===#
            for batch_data_test, batch_labels_test in data_and_labels_test:
                batch_pred_test = NN(batch_data_test)
                batch_loss_test = tf.reduce_mean(data_loss(batch_pred_test, batch_labels_test, label_dimensions))
                batch_loss_test += sum(NN.losses)
                mean_loss_test(batch_loss_test)
                mean_accuracy_test(tf.reduce_mean(accuracy(batch_pred_test, batch_labels_test)))
            
            #=== Track Training Metrics, Weights and Gradients ===#
            with summary_writer.as_default():
                tf.summary.scalar('loss_training', mean_loss_train.result(), step=epoch)
                tf.summary.scalar('accuracy_training', mean_accuracy_train.result(), step=epoch)
                tf.summary.scalar('loss_validation', mean_loss_val.result(), step=epoch)
                tf.summary.scalar('accuracy_validation', mean_accuracy_val.result(), step=epoch)
                tf.summary.scalar('loss_test', mean_loss_test.result(), step=epoch)
                tf.summary.scalar('accuracy_test', mean_accuracy_test.result(), step=epoch)
                for w in NN.weights:
                    tf.summary.histogram(w.name, w, step=epoch)
                l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
                for gradient, variable in zip(gradients, NN.trainable_variables):
                    tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)
            
            #=== Update Storage Arrays ===#
            storage_array_loss = np.append(storage_array_loss, mean_loss_train.result())
            storage_array_accuracy = np.append(storage_array_accuracy, mean_accuracy_test.result())

            #=== Display Epoch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
            print('Training Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_train.result(), mean_accuracy_train.result()))
            print('Validation Set: Loss: %.3e, Accuracy: %.3f' %(mean_loss_val.result(), mean_accuracy_val.result()))
            print('Test Set: Loss: %.3e, Accuracy: %.3f\n' %(mean_loss_test.result(), mean_accuracy_test.result()))
            print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            start_time_epoch = time.time()   
            
            #=== Reset Metrics ===#
            loss_validation = mean_loss_val.result()
            mean_loss_train.reset_states()
            mean_loss_val.reset_states()
            mean_loss_test.reset_states()
            mean_accuracy_train.reset_states()
            mean_accuracy_val.reset_states()
            mean_accuracy_test.reset_states()
                   
        ########################################################
        #   Updating Architecture and Saving Current Metrics   #
        ########################################################  
        print('================================')
        print('     Extending Architecture     ')
        print('================================')          
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss'] = storage_array_loss
        metrics_dict['accuracy'] = storage_array_accuracy
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv', index=False)
        
        #=== Sparsify Weights of Trained Layer ===#
        if run_options.use_L1 == 1:
            relative_number_zeros = NN.sparsify_weights_and_get_relative_number_of_zeros(hyperp.node_TOL)
            print('Relative Number of Zeros for Last Layer: %d\n' %(relative_number_zeros))
            storage_array_relative_number_zeros = np.append(storage_array_relative_number_zeros, relative_number_zeros)
        
        #=== Saving Relative Number of Zero Elements ===#
            relative_number_zeros_dict = {}
            relative_number_zeros_dict['rel_zeros'] = storage_array_relative_number_zeros
            df_relative_number_zeros = pd.DataFrame(relative_number_zeros_dict)
            df_relative_number_zeros.to_csv(file_paths.NN_savefile_name + "_relzeros" + '.csv', index=False)
       
        #=== Add Layer ===#
        trainable_hidden_layer_index += 1
        NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)
            
        #=== Preparing for Next Training Cycle ===#
        storage_array_loss = []
        storage_array_accuracy = []
        reset_optimizer   
        
    ########################
    #   Save Final Model   #
    ########################            
    #=== Saving Trained Model ===#          
    NN.save_weights(file_paths.NN_savefile_name)
    print('Final Model Saved') 
        

    
