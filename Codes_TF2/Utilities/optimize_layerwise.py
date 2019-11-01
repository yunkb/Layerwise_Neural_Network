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
    storage_loss_array = []
    storage_accuracy_array = []
    
    #####################################
    #   Training Current Architecture   #
    #####################################
    while loss_validation > hyper_p.error_TOL and trainable_hidden_layer_index < hyper_p.max_hidden_layers:    
        print('Beginning Training')
        for epoch in range(hyper_p.num_epochs):
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(run_options.filename)
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
            start_time_epoch = time.time()   
        
        ########################################################
        #   Updating Architecture and Saving Current Metrics   #
        ########################################################            
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss'] = storage_loss_array
        metrics_dict['accuracy'] = storage_accuracy_array
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(run_options.NN_savefile_name + "_metrics_hl" + str(trainable_hidden_layer_index) + '.csv', index=False)
        
        #=== Preparing for Next Iteration ===#
        loss_validation = loss_val_batch_average.result()
        trainable_hidden_layer_index += 1
        storage_loss_array = []
        storage_accuracy_array = []
        
        #=== Add Layer ===#
        NN.add_layer(trainable_hidden_layer_index, freeze=True, add = True)
    
    ########################
    #   Save Final Model   #
    ########################            
    NN.save_weights(run_options.NN_savefile_name)
    print('Final Model Saved') 
        

    
