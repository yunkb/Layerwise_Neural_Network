#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:33:26 2019

@author: hwan
"""

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

        
def compute_batch_metric(sess, NN, metric_op, num_data, minibatches):
    metric_value = 0 # Accuracy to be accumulated
    
    for batch_num in range(0, len(minibatches)):
        data_batch = minibatches[batch_num][0].T
        labels_batch = minibatches[batch_num][1].T
        misfit_sum = sess.run(metric_op, feed_dict = {NN.data_tf: data_batch, NN.labels_tf: labels_batch})    
        
        metric_value += misfit_sum
        
    metric_value = float(metric_value) / num_data

    return metric_value