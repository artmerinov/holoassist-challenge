#!/bin/bash

# copy files
rsync -av \
    train_action.py \
    eval_action.py \
    test_action.py \
    tsne.py \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist

# copy tsne_run_scripts/ folder
rsync -av tsne_run_scripts/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/tsne_run_scripts

# copy scr/ folder
rsync -av --exclude '__pycache__' src/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/src

# copy process_data/ folder
rsync -av --exclude '__pycache__' process_data/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/process_data

# copy run_scripts/ folder
rsync -av --exclude '__pycache__' run_scripts/ \
    amerinov@sdcslm01:/data/users/amerinov/projects/holoassist/run_scripts
