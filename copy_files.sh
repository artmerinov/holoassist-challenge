#!/bin/bash

# copy files
rsync -av \
    run.batch \
    train_action.py \
    train_action.sh \
    eval_action.py \
    train_mistake.py \
    train_mistake.sh \
    eval_mistake.py \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist

# copy scr/ folder
rsync -av --exclude '__pycache__' src/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/src