#!/bin/bash

# copy files
rsync -av \
    train_action.py \
    train_action.sh \
    train_action.batch \
    test_action.py \
    test_action.sh \
    test_action.batch \
    eval_action.py \
    eval_action.sh \
    eval_action.batch \
    train_mistake.py \
    train_mistake.sh \
    eval_mistake.py \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist

# copy scr/ folder
rsync -av --exclude '__pycache__' src/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/src

# copy process_data/ folder
rsync -av --exclude '__pycache__' process_data/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/process_data
