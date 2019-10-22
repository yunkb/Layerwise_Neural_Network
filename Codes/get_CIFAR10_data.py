#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:12:14 2019

@author: hwan
"""
import numpy as np
from CIFAR10_Hvass import cifar10
from CIFAR10_Hvass.cifar10 import img_size, num_channels, num_classes
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_CIFAR10_data():    
    cifar10.maybe_download_and_extract()    
    class_names = cifar10.load_class_names()
    data_train, cls_train, labels_train = cifar10.load_training_data()
    data_test, cls_test, labels_test = cifar10.load_test_data()
    
    num_training_data = len(data_train)
    num_testing_data = len(data_test)
    label_dimensions = num_classes 
        
    return num_training_data, num_testing_data, img_size, num_channels, label_dimensions, class_names, data_train, labels_train, data_test, labels_test

def get_CIFAR10_batch(data_train, labels_train, batch_size, flatten_data_flag):
    # Number of images in the training-set.
    num_images = len(data_train)

    # Create a random index.
    idx = np.random.choice(num_images, size=batch_size, replace=False)

    # Use the random index to select random images and labels.
    data_train_batch = data_train[idx, :, :, :]
    labels_train_batch = labels_train[idx, :]
    
    if flatten_data_flag == 1:
        data_train_batch = data_train_batch.reshape((batch_size, img_size*img_size*num_channels))

    return data_train_batch, labels_train_batch