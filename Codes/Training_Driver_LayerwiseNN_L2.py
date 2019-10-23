#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:43:31 2019

@author: hwan
"""

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR) # Suppresses all the messages when run begins
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from NN_FC_Layerwise import Layerwise
from get_MNIST_data import load_MNIST_data, get_MNIST_batch
from get_CIFAR10_data import load_CIFAR10_data, get_CIFAR10_batch
from save_trained_parameters import save_weights_and_biases

import time
import shutil # for deleting directories
from decimal import Decimal # for filenames

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    max_hidden_layers = 4
    error_TOL         = 1e-2
    batch_size        = 1000
    num_epochs        = 10
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p):   
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
        
        #=== Choose Data Set ===#
        self.data_MNIST = 0
        self.data_CIFAR10 = 1    
        
        #=== Setting Filename ===#   
        if self.data_MNIST == 1:
            data_type = 'MNIST'
        if self.data_CIFAR10 == 1:
            data_type = 'CIFAR10'
        
        #=== Filename ===#
        error_TOL_string = str('%.2e' %Decimal(hyper_p.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        self.filename = data_type + '_L2_mhl%d_eTOL%s_b%d_e%d' %(hyper_p.max_hidden_layers, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Saving neural network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we save the parameters for each layer separately, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the saved parameters

        #=== Creating Directories ===#
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyper_p, run_options):
    
    #=== Load Train and Test Data ===# 
    if run_options.data_MNIST == 1:
        mnist, num_training_data, num_testing_data, img_size, num_channels, data_dimensions, label_dimensions, data_test, labels_test = load_MNIST_data()
    if run_options.data_CIFAR10 == 1:    
        num_training_data, num_testing_data, img_size, num_channels, data_dimensions, label_dimensions, class_names, data_train, labels_train, data_test, labels_test = load_CIFAR10_data()   
        
    loss_value = 1e5
    hidden_layer_counter = 1
    
    while loss_value > hyper_p.error_TOL and hidden_layer_counter < hyper_p.max_hidden_layers:    
        ###########################
        #   Training Properties   #
        ###########################   
        #=== Neural network ===#
        NN = Layerwise(hyper_p, data_dimensions, label_dimensions, hidden_layer_counter, run_options.NN_savefile_name)
        
        #=== Loss functional ===#
        with tf.variable_scope('loss') as scope:
            loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = NN.prediction, labels = NN.labels_tf) )    
            tf.summary.scalar("loss",loss)
            
        #=== Relative Error ===#
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
                
        ###########################
        #   Train Neural Network  #
        ###########################          
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.initialize_all_variables()) 
            writer.add_graph(sess.graph)            
            
            #=== Train neural network ===#
            print('Beginning Training\n')
            start_time = time.time()
            num_batches = int(num_training_data/hyper_p.batch_size)
            for epoch in range(hyper_p.num_epochs):
                for batch_num in range(num_batches):
                    if run_options.data_MNIST == 1:
                        data_train_batch, labels_train_batch = get_MNIST_batch(mnist, hyper_p.batch_size)
                    if run_options.data_CIFAR10 == 1: 
                        data_train_batch, labels_train_batch = get_CIFAR10_batch(data_train, labels_train, hyper_p.batch_size)                                                   
                    sess.run(optimizer_Adam_op, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch}) 
                
                #=== Display Iteration Information ===#
                elapsed = time.time() - start_time
                loss_value = sess.run(loss, feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch}) 
                accuracy, s = sess.run([test_accuracy, summ], feed_dict = {NN.data_tf: data_test, NN.labels_tf: labels_test}) 
                writer.add_summary(s, epoch)
                print(run_options.filename)
                print('GPU: ' + hyper_p.gpu)
                print('Hidden Layers: %d, Epoch: %d, Loss: %.3e, Time: %.2f' %(hidden_layer_counter, epoch, loss_value, elapsed))
                print('Accuracy: %.2f\n' %(accuracy))
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
                print('Accuracy: %.2f\n' %(accuracy))
           
            #=== Save Final Model ===#
            save_weights_and_biases(sess, hyper_p, hidden_layer_counter, run_options.NN_savefile_name, 0)
            print('Final Model Saved')  
            
            #=== Close Session and Reset Graph ===#
            sess.close() 
        tf.reset_default_graph()
        hidden_layer_counter += 1
            

    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
            hyper_p.max_hidden_layers = int(sys.argv[1])
            hyper_p.error_TOL         = float(sys.argv[2])
            hyper_p.batch_size        = int(sys.argv[3])
            hyper_p.num_epochs        = int(sys.argv[4])
            hyper_p.gpu               = str(sys.argv[5])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     