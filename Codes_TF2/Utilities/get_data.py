#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:46:02 2019

@author: hwan
"""
import tensorflow as tf
from tensorflow.keras import datasets, utils
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_data(dataset, batch_size, random_seed):
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
    
    #=== Define Outputs ===#
    data_input_shape = data_train.shape[1:]
    num_channels = data_train.shape[-1]
    
    #=== Shuffling Data ===#
    data_and_labels_train_full = tf.data.Dataset.from_tensor_slices((data_train, labels_train)).shuffle(8192, seed=random_seed)
    data_and_labels_test = tf.data.Dataset.from_tensor_slices((data_test, labels_test)).shuffle(8192, seed=random_seed).batch(batch_size)
    
    #=== Partitioning Out Validation Set and Constructing Batches ===#
    num_training_data = int(0.8 * len(data_train))
    data_and_labels_train = data_and_labels_train_full.take(num_training_data).batch(batch_size)
    data_and_labels_val = data_and_labels_train_full.skip(num_training_data).batch(batch_size)    
    num_batches_train = len(list(data_and_labels_train))
    num_batches_val = len(list(data_and_labels_train))

    return data_and_labels_train, data_and_labels_test, data_and_labels_val, data_input_shape, num_channels, label_dimensions, num_batches_train, num_batches_val