#!/bin/bash

screen -S test -d -m
screen -S test . /workspace/hwan/anaconda3/etc/profile.d/conda.sh && conda activate tf_gpu && chdir Codes && ./Training_Driver_NNLayerwise_L2.py
