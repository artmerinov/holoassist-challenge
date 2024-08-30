#!/bin/bash

# copy files
rsync -av \
    train_action.py \
    train_GSF_InceptionV3_ImageNet_bs32_ns8.batch \
    train_GSF_InceptionV3_ImageNet_bs32_ns8.sh \
    train_GSF_InceptionV3_ImageNet_bs32_ns16.batch \
    train_GSF_InceptionV3_ImageNet_bs32_ns16.sh \
    train_GSF_InceptionV3_SS1_bs32_ns16.batch \
    train_GSF_InceptionV3_SS1_bs32_ns16.sh \
    train_GSF_ResNet50_ImageNet_bs32_ns8.batch \
    train_GSF_ResNet50_ImageNet_bs32_ns8.sh \
    train_GSF_ResNet50_SS1_bs32_ns12.batch \
    train_GSF_ResNet50_SS1_bs32_ns12.sh \
    train_GSM_InceptionV3_SS1_bs32_ns16.batch \
    train_GSM_InceptionV3_SS1_bs32_ns16.sh \
    train_TSM_InceptionV3_SS1_bs32_ns16.batch \
    train_TSM_InceptionV3_SS1_bs32_ns16.sh \
    train_TSN_InceptionV3_SS1_bs32_ns16.batch \
    train_TSN_InceptionV3_SS1_bs32_ns16.sh \
    eval_action.py \
    eval_GSF_InceptionV3_ImageNet_bs32_ns8.batch \
    eval_GSF_InceptionV3_ImageNet_bs32_ns8.sh \
    eval_GSF_InceptionV3_ImageNet_bs32_ns16.batch \
    eval_GSF_InceptionV3_ImageNet_bs32_ns16.sh \
    eval_GSF_InceptionV3_SS1_bs32_ns16.batch \
    eval_GSF_InceptionV3_SS1_bs32_ns16.sh \
    eval_GSF_ResNet50_ImageNet_bs32_ns8.batch \
    eval_GSF_ResNet50_ImageNet_bs32_ns8.sh \
    eval_GSF_ResNet50_SS1_bs32_ns12.batch \
    eval_GSF_ResNet50_SS1_bs32_ns12.sh \
    eval_GSM_InceptionV3_SS1_bs32_ns16.batch \
    eval_GSM_InceptionV3_SS1_bs32_ns16.sh \
    eval_TSM_InceptionV3_SS1_bs32_ns16.batch \
    eval_TSM_InceptionV3_SS1_bs32_ns16.sh \
    eval_TSN_InceptionV3_SS1_bs32_ns16.batch \
    eval_TSN_InceptionV3_SS1_bs32_ns16.sh \
    test_action.py \
    test_GSF_InceptionV3_ImageNet_bs32_ns16.batch \
    test_GSF_InceptionV3_ImageNet_bs32_ns16.sh \
    test_GSF_InceptionV3_SS1_bs32_ns16.batch \
    test_GSF_InceptionV3_SS1_bs32_ns16.sh \
    test_GSF_ResNet50_SS1_bs32_ns12.batch \
    test_GSF_ResNet50_SS1_bs32_ns12.sh \
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
