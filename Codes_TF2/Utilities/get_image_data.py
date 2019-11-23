#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:46:02 2019

@author: hwan
"""
import tensorflow as tf
from tensorflow.keras import datasets

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_data(NN_type, dataset, batch_size, random_seed):
    #=== Load Data ==#
    if dataset == 'MNIST':
        (data_train, labels_train), (data_test, labels_test) = datasets.mnist.load_data()
        data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
        data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
        label_dimensions = 10
    if dataset == 'CIFAR10':
        (data_train, labels_train), (data_test, labels_test) = datasets.cifar10.load_data()
        label_dimensions = 10
    if dataset == 'CIFAR100':
        (data_train, labels_train), (data_test, labels_test) = datasets.cifar100.load_data()
        label_dimensions = 100
    
    #=== Casting as float32 ===#
    data_train = tf.cast(data_train,tf.float32)
    labels_train = tf.cast(labels_train, tf.int32)
    data_test = tf.cast(data_test, tf.float32)
    labels_test = tf.cast(labels_test, tf.int32)
    
    #=== Normalize Data ===#
    data_train, data_test = data_train/255.0, data_test/255.0
    data_train = tf.image.per_image_standardization(data_train) # Linearly scales each image to have mean 0 and variance 1
    data_test = tf.image.per_image_standardization(data_test)   # Linearly scales each image to have mean 0 and variance 1
    
    #=== Flattening Image for Fully Connected Network ===#
    if NN_type == 'FC':
        if dataset == 'MNIST':
            data_train = tf.reshape(data_train, (len(data_train), 28*28))
            data_test = tf.reshape(data_test, (len(data_test), 28*28))
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            data_train = tf.reshape(data_train, (len(data_train), 32*32*3))
            data_test = tf.reshape(data_test, (len(data_test), 32*32*3))
    
    #=== Define Outputs ===#
    data_input_shape = data_train.shape[1:]
    if NN_type == 'CNN':
        num_channels = data_train.shape[-1]
    else:
        num_channels = None

    return data_train, labels_train, data_test, labels_test, data_input_shape, num_channels, label_dimensions