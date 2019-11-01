#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:55:15 2019

@author: hwan
"""

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL) # Suppresses all the messages when run begins
import numpy as np
import pandas as pd


A = tf.get_variable("array_test", dtype = tf.float32, shape = [2, 2], initializer = tf.random_normal_initializer(), trainable = False)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) 
    eval_A = sess.run("array_test" + ':0')
    data_A = {"A_test": eval_A.flatten()}
    df_A = pd.DataFrame(data_A)
    df_A.to_csv("array_data" + '.csv', index=False)
        
    restored_df_A = pd.read_csv("array_data" + '.csv', dtype = np.float32)
    restored_A = restored_df_A.values.reshape((2,2))