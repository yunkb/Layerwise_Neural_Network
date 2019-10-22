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
    testing_data = mnist.test.images
    testing_labels = mnist.test.labels
   
    return mnist, data_dimensions, label_dimensions, num_training_data, testing_data, testing_labels

