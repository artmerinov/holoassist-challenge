#!/bin/bash

# copy files
rsync -av \
    train_ae_with_memory.py \
    train_ae_with_memory.sh \
    eval_ae_with_memory.py \
    amerinov@basler:/data/amerinov/projects/holoassist

# copy scr/ folder
rsync -av --exclude '__pycache__' src/ \
    amerinov@basler:/data/amerinov/projects/holoassist/src

rsync -av logs/ \
    amerinov@basler:/data/amerinov/projects/holoassist/logs