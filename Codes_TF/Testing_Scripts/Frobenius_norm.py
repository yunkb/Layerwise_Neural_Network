#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:51:43 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

A = tf.constant(np.array([[1,0],[3,4]]), dtype = tf.float32)
fro_A_tf = tf.pow(tf.norm(A, 2), 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fro_A = sess.run(fro_A_tf)