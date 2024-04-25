#!/bin/bash

nohup python3 -B train_ae_with_memory.py | tee -a logs/train.log &
# python3 -B train_ae_with_memory.py