#!/bin/bash

# copy files
rsync -av \
    train_action.py \
    train_action.sh \
    eval_action.py \
    train_mistake.py \
    train_mistake.sh \
    eval_mistake.py \
    amerinov@basler:/data/amerinov/projects/holoassist

# copy scr/ folder
rsync -av --exclude '__pycache__' src/ \
    amerinov@basler:/data/amerinov/projects/holoassist/src

# rsync -av logs/ \
#     amerinov@basler:/data/amerinov/projects/holoassist/logs