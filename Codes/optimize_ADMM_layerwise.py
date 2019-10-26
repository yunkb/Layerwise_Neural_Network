#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:17:48 2019

@author: hwan
"""

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL) # Suppresses all the messages when run begins
import numpy as np

import shutil # for deleting directories
import os
import time

from random_mini_batches import random_mini_batches
from compute_batch_metric import compute_batch_metric
from save_trained_parameters_layerwise import save_weights_and_biases_FC, save_weights_and_biases_CNN

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def optimize_ADMM_layerwise(hyper_p, run_options, trainable_hidden_layer_index, NN, num_training_data, num_testing_data, pen, z_weights, z_biases, lagrange_weights, lagrange_biases, data_train, labels_train, data_test, labels_test):
###############################################################################
#                             Training Properties                             #
###############################################################################    
    #=== Loss functional ===#
    with tf.variable_scope('loss') as scope:
        data_loss_softmax_xent = tf.nn.softmax_cross_entropy_with_logits(logits = NN.logits, labels = NN.labels_tf)
        data_loss = tf.reduce_mean(data_loss_softmax_xent)    
        ADMM_penalty = 0.0
        for l in range(0, len(NN.weights)):  
            weights_norm = pen/2 * tf.pow(tf.norm(NN.weights[l] - z_weights[l] + lagrange_weights[l]/pen, 2), 2)
            biases_norm = pen/2 * tf.pow(tf.norm(NN.biases[l] - z_biases[l] + lagrange_biases[l]/pen, 2), 2)
            ADMM_penalty += weights_norm + biases_norm
        loss_functional = data_loss + ADMM_penalty
        data_loss_sum_softmax_xent = tf.reduce_sum(data_loss_softmax_xent)
        data_loss_train_accum_batch_tf = tf.placeholder(tf.float32, shape=())
        loss_train_accum_batch_tf = data_loss_train_accum_batch_tf + ADMM_penalty
        tf.summary.scalar("loss", loss_train_accum_batch_tf)
        
    #=== Accuracy ===#
    with tf.variable_scope('test_accuracy') as scope:
        minibatches_test = random_mini_batches(data_test.T, labels_test.T, hyper_p.batch_size, 1234)
        correct_tests_booleans = tf.equal(tf.argmax(tf.nn.softmax(NN.logits), 1), tf.argmax(NN.labels_tf, 1))
        accuracy_test_sum_correct_tests = tf.reduce_sum(tf.cast(correct_tests_booleans, 'float'))
        accuracy_test_accum_batch_tf = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar("test_accuracy", accuracy_test_accum_batch_tf)
                
    #=== Set optimizers ===#
    with tf.variable_scope('Training') as scope:
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(loss_functional,
                                                                 method='L-BFGS-B',
                                                                 options={'maxiter':10000,
                                                                          'maxfun':50000,
                                                                          'maxcor':50,
                                                                          'maxls':50,
                                                                          'ftol':1.0 * np.finfo(float).eps})
        #=== Track gradients ===#
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        gradients_tf = optimizer_Adam.compute_gradients(loss = loss_functional)
        for gradient, variable in gradients_tf:
            tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient))
        optimizer_Adam_op = optimizer_Adam.apply_gradients(gradients_tf)
                    
    #=== Set GPU configuration options ===#
    gpu_options = tf.GPUOptions(visible_device_list=hyper_p.gpu,
                                allow_growth=True)
    
    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True,
                                intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=2,
                                gpu_options= gpu_options)
    
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    summ = tf.summary.merge_all()
    if os.path.exists('../Tensorboard/' + run_options.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + run_options.filename)  
    writer = tf.summary.FileWriter('../Tensorboard/' + run_options.filename)    
    
###############################################################################
#                          Train Neural Network                               #
###############################################################################
    storage_loss_array = np.array([])
    storage_accuracy_array = np.array([])
             
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.initialize_all_variables()) 
        writer.add_graph(sess.graph)
        
        #=== Assign initial value of z to be equal to w ===#
        for l in range(0, len(NN.weights)): 
            sess.run("z_weights_initial_value" + str(l+1)) 
            sess.run("z_biases_initial_value" + str(l+1))  
        
        #=== Train neural network ===#
        print('Beginning Training\n')
        num_batches = int(num_training_data/hyper_p.batch_size)
        for epoch in range(hyper_p.num_epochs):           
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(run_options.filename)
            print('Trainable Hidden Layer Index: %d' %trainable_hidden_layer_index)
            print('GPU: ' + hyper_p.gpu + '\n')
            print('Optimizing %d batches of size %d:' %(num_batches, hyper_p.batch_size))
            start_time_epoch = time.time()
            minibatches_train = random_mini_batches(data_train.T, labels_train.T, hyper_p.batch_size, 1234)
            for batch_num in range(num_batches):
                data_train_batch = minibatches_train[batch_num][0].T
                labels_train_batch = minibatches_train[batch_num][1].T
                start_time_batch = time.time()
                sess.run(optimizer_Adam_op, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch})
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
            
            #=== Display Batch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            current_loss = compute_batch_metric(sess, NN, data_loss_sum_softmax_xent, num_training_data, minibatches_train)
            current_accuracy = compute_batch_metric(sess, NN, accuracy_test_sum_correct_tests, num_testing_data, minibatches_test)
            storage_loss_array = np.append(storage_loss_array, current_loss)
            storage_accuracy_array = np.append(storage_accuracy_array, current_accuracy)
            s = sess.run(summ, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch, loss_train_accum_batch_tf: current_loss, accuracy_test_accum_batch_tf: current_accuracy}) 
            writer.add_summary(s, epoch)
            print('Time per Epoch: %.2f' %(elapsed_time_epoch))
            print('Loss: %.3e, Accuracy: %.3f\n' %(current_loss, current_accuracy))
            start_time_epoch = time.time()   
                 
            #=== Optimize with LBFGS ===#
            if run_options.use_LBFGS == 1:
                print('Optimizing with LBFGS:')   
                start_time_LBFGS = time.time()
                optimizer_LBFGS.minimize(sess, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch})
                time_elapsed_LBFGS = time.time() - start_time_LBFGS 
                current_loss = compute_batch_metric(sess, NN, data_loss_sum_softmax_xent, num_training_data, minibatches_train)
                current_accuracy = compute_batch_metric(sess, NN, accuracy_test_sum_correct_tests, num_testing_data, minibatches_test)
                storage_loss_array = np.append(storage_loss_array, current_loss)
                storage_accuracy_array = np.append(storage_accuracy_array, current_accuracy)
                s = sess.run(summ, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch, loss_train_accum_batch_tf: current_loss, accuracy_test_accum_batch_tf: current_accuracy}) 
                writer.add_summary(s, epoch)
                print('LBFGS Optimization Complete')
                print('Time for LBFGS: %.2f' %(time_elapsed_LBFGS))
                print('Loss: %.3e, Accuracy: %.3f\n' %(current_loss, current_accuracy)) 
                
            #=== Update z and Lagrange Multiplier ===# 
            for l in range(0, len(NN.weights)):  
                sess.run("z_weights_update" + str(l+1))
                sess.run("z_biases_update" + str(l+1))
                sess.run("lagrange_weights_update" + str(l+1))
                sess.run("lagrange_biases_update" + str(l+1))
        
        #=== Save Final Model ===#
        if run_options.NN_type == 'FC':
            relative_number_zeros = save_weights_and_biases_FC(sess, hyper_p, trainable_hidden_layer_index, run_options.NN_savefile_name, thresholding_flag = 1)
        if run_options.NN_type == 'CNN':
            relative_number_zeros = save_weights_and_biases_CNN(sess, hyper_p, trainable_hidden_layer_index, run_options.NN_savefile_name, thresholding_flag = 1)
        print('Final Model Saved')  
        
        #=== Close Session ===#
        sess.close() 
        
        return storage_loss_array, storage_accuracy_array, relative_number_zeros