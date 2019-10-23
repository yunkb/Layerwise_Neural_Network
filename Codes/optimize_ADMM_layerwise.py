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
from save_trained_parameters_layerwise import save_weights_and_biases

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def optimize_ADMM_layerwise(hyper_p, run_options, hidden_layer_counter, NN, num_training_data, pen, z_weights, z_biases, lagrange_weights, lagrange_biases, data_train, labels_train, data_test, labels_test):
###############################################################################
#                             Training Properties                             #
###############################################################################    
    #=== Loss functional ===#
    with tf.variable_scope('loss') as scope:
        data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = NN.prediction, labels = NN.labels_tf))    
        ADMM_penalty = 0.0
        for l in range(0, len(NN.weights)):  
            weights_norm = pen/2 * tf.pow(tf.norm(NN.weights[l] - z_weights[l] + lagrange_weights[l]/pen, 2), 2)
            biases_norm = pen/2 * tf.pow(tf.norm(NN.biases[l] - z_biases[l] + lagrange_biases[l]/pen, 2), 2)
            ADMM_penalty += weights_norm + biases_norm
        loss = data_loss + ADMM_penalty
        tf.summary.scalar("loss",loss)
        
    #=== Accuracy ===#
    with tf.variable_scope('test_accuracy') as scope:
        num_correct_tests = tf.equal(tf.argmax(NN.prediction, 1), tf.argmax(NN.labels_tf, 1))
        test_accuracy = tf.reduce_mean(tf.cast(num_correct_tests, 'float'))
        tf.summary.scalar("test_accuracy", test_accuracy)
                
    #=== Set optimizers ===#
    with tf.variable_scope('Training') as scope:
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                                 method='L-BFGS-B',
                                                                 options={'maxiter':10000,
                                                                          'maxfun':50000,
                                                                          'maxcor':50,
                                                                          'maxls':50,
                                                                          'ftol':1.0 * np.finfo(float).eps})
        #=== Track gradients ===#
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        gradients_tf = optimizer_Adam.compute_gradients(loss = loss)
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
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.initialize_all_variables()) 
        writer.add_graph(sess.graph)
        
        #=== Assign initial value of z to be equal to w ===#
        for l in range(0, len(NN.weights)): 
            sess.run("z_weights_initial_value" + str(l+1)) 
            sess.run("z_biases_initial_value" + str(l+1))  
        
        #=== Train neural network ===#
        print('Beginning Training\n')
        start_time = time.time()
        num_batches = int(num_training_data/hyper_p.batch_size)
        for epoch in range(hyper_p.num_epochs):
            minibatches = random_mini_batches(data_train.T, labels_train.T, hyper_p.batch_size, 1234)
            for batch_num in range(num_batches):
                data_train_batch = minibatches[batch_num][0].T
                labels_train_batch = minibatches[batch_num][1].T
                sess.run(optimizer_Adam_op, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch})
            
            #=== Display Iteration Information ===#    
            elapsed = time.time() - start_time
            loss_value = sess.run(loss, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch}) 
            accuracy, s = sess.run([test_accuracy, summ], feed_dict = {NN.data_tf: data_test, NN.labels_tf: labels_test}) 
            writer.add_summary(s, epoch)
            print(run_options.filename)
            print('GPU: ' + hyper_p.gpu)
            print('Hidden Layers: %d, Epoch: %d, Loss: %.3e, Time: %.2f' %(hidden_layer_counter, epoch, loss_value, elapsed))
            print('Accuracy: %.3f\n' %(accuracy))
            start_time = time.time() 
            
            #=== Optimize with LBFGS ===#
            if run_options.use_LBFGS == 1:
                print('Optimizing with LBFGS')   
                optimizer_LBFGS.minimize(sess, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch})
                elapsed = time.time() - start_time 
                loss_value = sess.run(loss, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch})
                accuracy, s = sess.run([test_accuracy, summ], feed_dict = {NN.data_tf: data_test, NN.labels_tf: labels_test}) 
                writer.add_summary(s, epoch)
                print('LBFGS Optimization Complete')
                print('Loss: %.3e, Time: %.2f' %(loss_value, elapsed))
                print('Accuracy: %.3f\n' %(accuracy))
            
            #=== Update z and Lagrange Multiplier ===# 
            for l in range(0, len(NN.weights)):  
                sess.run("z_weights_update" + str(l+1))
                sess.run("z_biases_update" + str(l+1))
                sess.run("lagrange_weights_update" + str(l+1))
                sess.run("lagrange_biases_update" + str(l+1))
        
        #=== Save Final Model ===#
        save_weights_and_biases(sess, hyper_p, hidden_layer_counter, run_options.NN_savefile_name, thresholding_flag = 1)
        print('Final Model Saved')  

        #=== Close Session ===#
        sess.close() 