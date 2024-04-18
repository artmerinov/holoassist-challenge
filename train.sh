#!/bin/bash

nohup python3 -B train.py \
    --holoassist_dir /data/amerinov/data/holoassist \
    --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
    --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --dataset_name holoassist \
    --base_model InceptionV3 \
    --fusion_mode GSF \
    --dropout 0.5 \
    --num_epochs 10 \
    --batch_size 16 \
    --num_segments 8 \
    --lr 0.01 \
    --clip_gradient 20 \
    | tee -a logs/train.log &