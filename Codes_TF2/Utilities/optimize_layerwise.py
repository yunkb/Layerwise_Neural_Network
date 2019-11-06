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
#                             Loss and Accuracy                               #
###############################################################################
def data_loss(y_pred, y_true, label_dimensions):
    y_true = tf.one_hot(tf.cast(y_true,tf.int64), label_dimensions, dtype=tf.float32)
    return  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred))

def accuracy(y_pred,y_true):
    correct = tf.math.in_top_k(tf.cast(tf.squeeze(y_true),tf.int64),tf.cast(y_pred, tf.float32),  1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyper_p, run_options, NN, data_and_labels_train, data_and_labels_test, data_and_labels_val, label_dimensions, num_batches_train):
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()
    reset_optimizer = tf.group([v.initializer for v in optimizer.variables()])
    
    #=== Define Metrics and Initialize Metric Storage Arrays ===#
    loss_train_batch_average = tf.keras.metrics.Mean()
    loss_val_batch_average = tf.keras.metrics.Mean()
    accuracy_train_batch_average = tf.keras.metrics.Mean()
    accuracy_val_batch_average = tf.keras.metrics.Mean()
    storage_loss_array = np.array([])
    storage_accuracy_array = np.array([])
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists('../Tensorboard/' + run_options.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + run_options.filename)  
    summary_writer = tf.summary.create_file_writer('../Tensorboard/' + run_options.filename)

###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    loss_validation = 1e5
    trainable_hidden_layer_index = 2
    relative_number_zeros = 0
    retrain = 0
    storage_loss_array = []
    storage_accuracy_array = []
    storage_relative_number_zeros_array = []
    
    #####################################
    #   Training Current Architecture   #
    #####################################
    while loss_validation > hyper_p.error_TOL and trainable_hidden_layer_index < hyper_p.max_hidden_layers:    
        #=== Initial Loss and Accuracy ===#
        for batch_num, (data_train, labels_train) in data_and_labels_train.enumerate():
            output = NN(data_train)
            loss_train_batch = data_loss(output, labels_train, label_dimensions)
            loss_train_batch += sum(NN.losses)
            loss_train_batch_average(loss_train_batch) 
            accuracy_train_batch_average(accuracy(output, labels_train))
        for data_val, labels_val in data_and_labels_val:
            output_val = NN(data_val)
            loss_val_batch = data_loss(output_val, labels_val, label_dimensions)
            loss_val_batch += sum(NN.losses)
            loss_val_batch_average(loss_val_batch)
            accuracy_val_batch_average(accuracy(output_val, labels_val))
        storage_loss_array = np.append(storage_loss_array, loss_train_batch_average.result())
        storage_accuracy_array = np.append(storage_accuracy_array, accuracy_val_batch_average.result())
        print('Initial Losses:')
        print('Training Set: Loss: %.3e, Accuracy: %.3f' %(loss_train_batch_average.result(), accuracy_train_batch_average.result()))
        print('Validation Set: Loss: %.3e, Accuracy: %.3f\n' %(loss_val_batch_average.result(), accuracy_val_batch_average.result()))
        
        #=== Begin Training ===#
        print('Beginning Training')
        for epoch in range(hyper_p.num_epochs):
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(run_options.filename)
            print('Trainable Hidden Layer Index: %d' %(trainable_hidden_layer_index))
            print('GPU: ' + hyper_p.gpu + '\n')
            print('Optimizing %d batches of size %d:' %(num_batches_train, hyper_p.batch_size))
            start_time_epoch = time.time()
            for batch_num, (data_train, labels_train) in data_and_labels_train.enumerate():
                with tf.GradientTape() as tape:
                    start_time_batch = time.time()
                    output = NN(data_train)
                    #=== Display Model Summary ===#
                    if batch_num == 0 and epoch == 0:
                        NN.summary()
                    loss_train_batch = data_loss(output, labels_train, label_dimensions)
                    loss_train_batch += sum(NN.losses)
                    gradients = tape.gradient(loss_train_batch, NN.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                    elapsed_time_batch = time.time() - start_time_batch
                    if batch_num  == 0:
                        print('Time per Batch: %.2f' %(elapsed_time_batch))
                loss_train_batch_average(loss_train_batch) 
                accuracy_train_batch_average(accuracy(output, labels_train))
                                        
            #=== Computing Accuracy ===#
            for data_val, labels_val in data_and_labels_val:
                output_val = NN(data_val)
                loss_val_batch = data_loss(output_val, labels_val, label_dimensions)
                loss_val_batch += sum(NN.losses)
                loss_val_batch_average(loss_val_batch)
                accuracy_val_batch_average(accuracy(output_val, labels_val))
            
            #=== Track Training Metrics, Weights and Gradients ===#
            with summary_writer.as_default():
                tf.summary.scalar('loss_training', loss_train_batch_average.result(), step=epoch)
                tf.summary.scalar('accuracy_training', accuracy_train_batch_average.result(), step=epoch)
                tf.summary.scalar('loss_validation', loss_val_batch_average.result(), step=epoch)
                tf.summary.scalar('accuracy_validation', accuracy_val_batch_average.result(), step=epoch)
                storage_loss_array = np.append(storage_loss_array, loss_train_batch_average.result())
                storage_accuracy_array = np.append(storage_accuracy_array, accuracy_val_batch_average.result())
                for w in NN.weights:
                    tf.summary.histogram(w.name, w, step=epoch)
                l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
                for gradient, variable in zip(gradients, NN.trainable_variables):
                    tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)
                
            #=== Display Epoch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
            print('Training Set: Loss: %.3e, Accuracy: %.3f' %(loss_train_batch_average.result(), accuracy_train_batch_average.result()))
            print('Validation Set: Loss: %.3e, Accuracy: %.3f\n' %(loss_val_batch_average.result(), accuracy_val_batch_average.result()))
            print('Previous Layer Relative # of 0s: %.7f\n' %(relative_number_zeros))
            if run_options.use_unfreeze_all_and_train == 1:    
                print('retrain equals %d' %(retrain))
            start_time_epoch = time.time()   
            
            #=== Reset Metrics ===#
            loss_validation = loss_val_batch_average.result()
            loss_train_batch_average.reset_states()
            loss_val_batch_average.reset_states()
            accuracy_train_batch_average.reset_states()
            accuracy_val_batch_average.reset_states()
                   
        ########################################################
        #   Updating Architecture and Saving Current Metrics   #
        ########################################################  
        print('================================')
        print('     Extending Architecture     ')
        print('================================')          
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss'] = storage_loss_array
        metrics_dict['accuracy'] = storage_accuracy_array
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(run_options.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv', index=False)
        
        #=== Sparsify Weights of Trained Layer ===#
        if run_options.use_L1 == 1:
            relative_number_zeros = NN.sparsify_weights_and_get_relative_number_of_zeros(hyper_p.node_TOL)
            print('Relative Number of Zeros for Last Layer: %d\n' %(relative_number_zeros))
            storage_relative_number_zeros_array = np.append(storage_relative_number_zeros_array, relative_number_zeros)
        
        #=== Saving Relative Number of Zero Elements ===#
            relative_number_zeros_dict = {}
            relative_number_zeros_dict['rel_zeros'] = storage_relative_number_zeros_array
            df_relative_number_zeros = pd.DataFrame(relative_number_zeros_dict)
            df_relative_number_zeros.to_csv(run_options.NN_savefile_name + "_relzeros" + '.csv', index=False)
       
        #=== Add Layer ===#
        if run_options.use_unfreeze_all_and_train == 1:                 
            if trainable_hidden_layer_index > 2 and retrain == 0:
                NN.add_layer(trainable_hidden_layer_index, freeze=False, add = False)
                retrain = 1
            elif trainable_hidden_layer_index == 2 or (trainable_hidden_layer_index > 2 and retrain == 1):
                trainable_hidden_layer_index += 1
                NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)
                retrain = 0
        else:
            trainable_hidden_layer_index += 1
            NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)
            if run_options.use_regularization_scheduler == 1:
                hyper_p.regularization += 0.0002
                NN.get_layer('W' + str(trainable_hidden_layer_index)).kernel_regularizer = tf.keras.regularizers.l1(hyper_p.regularization)
                NN.get_layer('W' + str(trainable_hidden_layer_index)).bias_regularizer = tf.keras.regularizers.l1(hyper_p.regularization)
            
        #=== Preparing for Next Training Cycle ===#
        storage_loss_array = []
        storage_accuracy_array = []
        reset_optimizer   
        
    ########################
    #   Save Final Model   #
    ########################            
    #=== Saving Trained Model ===#          
    NN.save_weights(run_options.NN_savefile_name)
    print('Final Model Saved') 
        

    
