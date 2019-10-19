#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:43:31 2019

@author: hwan
"""

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL) # Suppresses all the messages when run begins
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd

from NN_Layerwise import Layerwise
from random_mini_batches import random_mini_batches

import time
import shutil # for deleting directories

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

###############################################################################
#                       Hyperparameters and Filenames                         #
###############################################################################
class HyperParameters:
    max_num_hidden_layers = 3
    num_hidden_nodes  = 200
    penalty           = 0.1
    num_training_data = 20
    batch_size        = 20
    num_epochs        = 2000
    gpu               = '1'
    
class RunOptions:
    def __init__(self, hyper_p):          
        # Other options
        self.num_testing_data = 200
        
        # File name
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = hyper_p.data_type + '_hl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.max_num_hidden_layers, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

        # Saving neural network
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files

        # Creating Directories
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyper_p, run_options):
        
    hyper_p.batch_size = hyper_p.num_training_data
    
    # Load Train and Test Data  
    mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
     
    ###########################
    #   Training Properties   #
    ###########################   
    # Neural network
    NN = Layerwise(hyper_p, run_options, 1024, 10, construct_flag = 1)
    
    # Loss functional
    with tf.variable_scope('loss') as scope:
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = NN.labels_tf) )    
        tf.summary.scalar("loss",loss)
        
    # Relative Error
    with tf.variable_scope('test_accuracy') as scope:
        test_accuracy = tf.norm(NN.parameter_input_test_tf - NN.autoencoder_pred_test, 2)/tf.norm(NN.parameter_input_test_tf, 2)
        tf.summary.scalar("test_accuracy", test_accuracy)
                
    # Set optimizers
    with tf.variable_scope('Training') as scope:
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                                 method='L-BFGS-B',
                                                                 options={'maxiter':10000,
                                                                          'maxfun':50000,
                                                                          'maxcor':50,
                                                                          'maxls':50,
                                                                          'ftol':1.0 * np.finfo(float).eps})
        # Track gradients
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        gradients_tf = optimizer_Adam.compute_gradients(loss = loss)
        for gradient, variable in gradients_tf:
            tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient))
        optimizer_Adam_op = optimizer_Adam.apply_gradients(gradients_tf)
                    
    # Set GPU configuration options
    gpu_options = tf.GPUOptions(visible_device_list=hyper_p.gpu,
                                allow_growth=True)
    
    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True,
                                intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=2,
                                gpu_options= gpu_options)
    
    # Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    summ = tf.summary.merge_all()
    if os.path.exists('../Tensorboard/' + run_options.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + run_options.filename)  
    writer = tf.summary.FileWriter('../Tensorboard/' + run_options.filename)
    
    # Saver for saving trained neural network
    saver = tf.train.Saver(NN.saver_autoencoder)
    
    ########################
    #   Train Autoencoder  #
    ########################          
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.initialize_all_variables()) 
        writer.add_graph(sess.graph)
        
        # Save neural network
        saver.save(sess, run_options.NN_savefile_name)
        
        # Train neural network
        print('Beginning Training\n')
        start_time = time.time()
        num_batches = int(hyper_p.num_training_data/hyper_p.batch_size)
        for epoch in range(hyper_p.num_epochs):
            if num_batches == 1:
                tf_dict = {NN.parameter_input_tf: parameter_train, NN.state_obs_tf: state_obs_train,
                           NN.parameter_input_test_tf: parameter_test, NN.state_obs_test_tf: state_obs_test, NN.state_obs_inverse_input_tf: state_obs_test} 
                loss_value, _, s = sess.run([loss, optimizer_Adam_op, summ], tf_dict)  
                if run_options.check_gradients == 1:
                    print('Checking gradient')
                    check_gradients_directional_derivative(sess, NN, loss, gradients_tf, tf_dict)
                    pdb.set_trace()
                writer.add_summary(s, epoch)
            else:
                minibatches = random_mini_batches(parameter_train.T, state_obs_train.T, hyper_p.batch_size, 1234)
                for batch_num in range(num_batches):
                    parameter_train_batch = minibatches[batch_num][0].T
                    state_obs_train_batch = minibatches[batch_num][1].T
                    tf_dict = {NN.parameter_input_tf: parameter_train_batch, NN.state_obs_tf: state_obs_train_batch} 
                    loss_value, _, s = sess.run([loss, optimizer_Adam_op, summ], tf_dict) 
                    writer.add_summary(s, epoch)
                
            # print to monitor results
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                print(run_options.filename)
                print('GPU: ' + hyper_p.gpu)
                print('Epoch: %d, Loss: %.3e, Time: %.2f\n' %(epoch, loss_value, elapsed))
                start_time = time.time()     
                
            # save every 1000 epochs
            if epoch % 1000 == 0:
                saver.save(sess, run_options.NN_savefile_name, write_meta_graph=False)
                 
        # Optimize with LBFGS
        print('Optimizing with LBFGS\n')   
        optimizer_LBFGS.minimize(sess, feed_dict=tf_dict)
        [loss_value, s] = sess.run([loss,summ], tf_dict)
        writer.add_summary(s,hyper_p.num_epochs)
        print('LBFGS Optimization Complete\n') 
        elapsed = time.time() - start_time
        print('Loss: %.3e, Time: %.2f\n' %(loss_value, elapsed))
        
        # Save final model
        saver.save(sess, run_options.NN_savefile_name, write_meta_graph=False)   
        print('Final Model Saved')  
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    # Hyperparameters    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
            hyper_p.max_num_hidden_layers = int(sys.argv[2])
            hyper_p.num_hidden_nodes  = int(sys.argv[4])
            hyper_p.penalty           = float(sys.argv[5])
            hyper_p.num_training_data = int(sys.argv[6])
            hyper_p.batch_size        = int(sys.argv[7])
            hyper_p.num_epochs        = int(sys.argv[8])
            hyper_p.gpu               = str(sys.argv[9])
            
    # Set run options         
    run_options = RunOptions(hyper_p)
    
    # Initiate training
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     