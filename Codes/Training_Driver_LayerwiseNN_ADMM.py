#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:43:31 2019

@author: hwan
"""

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL) # Suppresses all the messages when run begins
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from NN_Layerwise import Layerwise
from save_trained_parameters import save_weights_and_biases
from get_MNIST_data import load_MNIST_data, get_MNIST_batch
from get_CIFAR10_data import load_CIFAR10_data, get_CIFAR10_batch
from ADMM_methods import construct_ADMM_objects, ADMM_penalty_term, update_z_and_lagrange_multiplier_tf_operations, update_z_and_lagrange_multiplier

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
tf.set_random_seed(1234)

###############################################################################
#                       Hyperparameters and RunOptions                        #
###############################################################################
class HyperParameters:
    regularization    = 0.01
    penalty           = 1
    node_TOL          = 1e-3
    error_TOL         = 1e-3
    batch_size        = 100
    num_epochs        = 5
    gpu               = '1'
    
class RunOptions:
    def __init__(self, hyper_p): 
        self.data_MNIST = 0
        self.data_CIFAR10 = 1    
        if self.data_MNIST == 1:
            data_type = 'MNIST'
        if self.data_CIFAR10 == 1:
            data_type = 'CIFAR10'
        
        #=== File name ===#
        if hyper_p.regularization >= 1:
            hyper_p.regularization = int(hyper_p.regularization)
            regularization_string = str(hyper_p.regularization)
        else:
            regularization_string = str(hyper_p.regularization)
            regularization_string = 'pt' + regularization_string[2:]
            
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]
            
        node_TOL_string = str('%.2e' %Decimal(hyper_p.node_TOL))
        node_TOL_string = node_TOL_string[-1]
        error_TOL_string = str('%.2e' %Decimal(hyper_p.error_TOL))
        error_TOL_string = error_TOL_string[-1]
        
        self.filename = data_type + '_ADMM_r%s_p%s_nTOL%s_eTOL%s_b%d_e%d' %(regularization_string, penalty_string, node_TOL_string, error_TOL_string, hyper_p.batch_size, hyper_p.num_epochs)

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
        mnist, num_training_data, num_testing_data, data_dimensions, label_dimensions, data_test, labels_test = load_MNIST_data()
    if run_options.data_CIFAR10 == 1:    
        num_training_data, num_testing_data, img_size, num_channels, label_dimensions, class_names, data_train, labels_train, data_test, labels_test = load_CIFAR10_data()
        data_dimensions = img_size*img_size*num_channels
        
    #=== Iteration Objects ===#
    loss_value = 1e5
    weight_list_counter = 0
    
    while loss_value > hyper_p.error_TOL:     
        ###########################
        #   Training Properties   #
        ###########################   
        #=== Neural network ===#
        NN = Layerwise(hyper_p, data_dimensions, label_dimensions, weight_list_counter, run_options.NN_savefile_name)
        
        #=== Initialize ADMM objects ===#
        z_weights, z_biases, lagrange_weights, lagrange_biases = construct_ADMM_objects(NN)
        alpha = tf.constant(hyper_p.regularization, dtype = tf.float32)
        pen = tf.constant(hyper_p.penalty, dtype = tf.float32)
        update_z_and_lagrange_multiplier_tf_operations(NN, alpha, pen, z_weights, z_biases, lagrange_weights, lagrange_biases)
    
        #=== Loss functional ===#
        with tf.variable_scope('loss') as scope:
            data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = NN.prediction, labels = NN.labels_tf))    
            ADMM_penalty = ADMM_penalty_term(NN, pen, z_weights, z_biases, lagrange_weights, lagrange_biases)       
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
        
        #=== Tensorboard ===# # Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
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
            
            #=== Assign initial value of z to be equal to w ===#
            for l in range(0, len(NN.weights)): 
                sess.run("z_weights_initial_value" + str(l+1)) 
                sess.run("z_biases_initial_value" + str(l+1))  
            
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
                    #loss_value, _, s = sess.run([loss, optimizer_Adam_op, summ], feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch}) 
                    #writer.add_summary(s, epoch)
                    loss_value, _ = sess.run([loss, optimizer_Adam_op], feed_dict = {NN.data_tf: data_train_batch, NN.labels_tf: labels_train_batch}) 
                update_z_and_lagrange_multiplier(sess, len(NN.weights))
                                
                #=== Display Iteration Information ===#
                elapsed = time.time() - start_time
                print(run_options.filename)
                print('GPU: ' + hyper_p.gpu)
                print('Hidden Layers: %d, Epoch: %d, Loss: %.3e, Time: %.2f' %(weight_list_counter+1, epoch, loss_value, elapsed))
                accuracy, s = sess.run([test_accuracy, summ], feed_dict = {NN.data_tf: data_test, NN.labels_tf: labels_test}) 
                writer.add_summary(s, epoch)
                #accuracy = sess.run(test_accuracy, feed_dict = {NN.data_tf: data_test, NN.labels_tf: labels_test}) 
                print('Accuracy: %.2f\n' %(accuracy))
                start_time = time.time()   
                   
            #=== Optimize with LBFGS ===#
    # =============================================================================
    #         print('Optimizing with LBFGS\n')   
    #         optimizer_LBFGS.minimize(sess, feed_dict=tf_dict)
    #         [loss_value, s] = sess.run([loss,summ], tf_dict)
    #         writer.add_summary(s,hyper_p.num_epochs)
    #         print('LBFGS Optimization Complete\n') 
    #         elapsed = time.time() - start_time
    #         print('Loss: %.3e, Time: %.2f\n' %(loss_value, elapsed))
    # =============================================================================
            
            #=== Save final model ===#
            save_weights_and_biases(sess, hyper_p, weight_list_counter, run_options.NN_savefile_name, 1)
            print('Final Model Saved')  
            
            #=== Network Predictions ===#
            index = 4389 # There are 55,000 training examples in MNIST
            mnist_digit = mnist.test.images[index]
            digit = np.array(mnist_digit, dtype='float')
            pixels = digit.reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.show()    
            print(sess.run(NN.classify, feed_dict={NN.data_tf: mnist_digit.reshape(1,784)}))
            
            #=== Reset Graph and Close Session ===#
            sess.close()
        tf.reset_default_graph()
        weight_list_counter += 1
    
###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
            hyper_p.regularization    = float(sys.argv[1])
            hyper_p.penalty           = float(sys.argv[2])
            hyper_p.node_TOL          = float(sys.argv[3])
            hyper_p.error_TOL         = float(sys.argv[4])
            hyper_p.batch_size        = int(sys.argv[5])
            hyper_p.num_epochs        = int(sys.argv[6])
            hyper_p.gpu               = str(sys.argv[7])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     