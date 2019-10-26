#!/bin/env bash

screen -S test -d -m
screen -S test -X source activate tf_gpu
screen -S test -X chdir Codes
screen -S test -X ./Training_Driver_NNLayerwise_L2.py
