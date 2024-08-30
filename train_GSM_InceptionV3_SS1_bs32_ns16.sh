#!/bin/bash

python3 -B /data/users/amerinov/projects/holoassist/train_action.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --raw_annotation_file /data/users/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/users/amerinov/data/holoassist/data-splits-v1_2 \
    --fga_map_file /data/users/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --num_classes 1887 \
    --base_model InceptionV3 \
    --fusion_mode GSM \
    --pretrained SS1 \
    --num_epochs 18 \
    --batch_size 32 \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_segments 16 \
    --lr 0.01 \
    --clip_gradient 20 \
    | tee -a /data/users/amerinov/projects/holoassist/logs/train_GSM_InceptionV3_SS1_bs32_ns16.log
