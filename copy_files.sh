#!/bin/bash

# copy files
rsync -av \
    train_action.py \
    train_action.sh \
    train_mistake.py \
    train_mistake.sh \
    amerinov@basler:/data/amerinov/projects/holoassist

# copy scr/ folder
rsync -av --exclude '__pycache__' src/ \
    amerinov@basler:/data/amerinov/projects/holoassist/src

# rsync -av logs/ \
#     amerinov@basler:/data/amerinov/projects/holoassist/logs