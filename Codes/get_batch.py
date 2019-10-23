#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:59:32 2019

@author: hwan
"""

import numpy as np

def get_batch(data_train, labels_train, batch_size):
    # Number of images in the training-set.
    num_images = len(data_train)

    # Create a random index.
    idx = np.random.choice(num_images, size=batch_size, replace=False)

    # Use the random index to select random images and labels.
    data_train_batch = data_train[idx, :]
    labels_train_batch = labels_train[idx, :]

    return data_train_batch, labels_train_batch