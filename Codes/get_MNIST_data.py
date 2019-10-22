#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:45:56 2019

@author: hwan
"""

from tensorflow.examples.tutorials.mnist import input_data
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_MNIST_data():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
    data_dimensions = mnist.test.images[0].shape[0]
    label_dimensions = mnist.test.labels[0].shape[0]
    num_training_data = mnist.train.num_examples
    num_testing_data = mnist.test.num_examples
    data_test = mnist.test.images
    labels_test = mnist.test.labels
   
    return mnist, num_training_data, num_testing_data, data_dimensions, label_dimensions, data_test, labels_test

def get_MNIST_batch(mnist, batch_size):
    data_train_batch, labels_train_batch = mnist.train.next_batch(batch_size)
    
    return data_train_batch, labels_train_batch