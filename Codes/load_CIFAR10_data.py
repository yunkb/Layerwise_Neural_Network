#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:12:14 2019

@author: hwan
"""
from CIFAR10_Hvass import cifar10
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_CIFAR10_data():    
    cifar10.maybe_download_and_extract()    
    class_names = cifar10.load_class_names()
    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()
    pdb.set_trace()