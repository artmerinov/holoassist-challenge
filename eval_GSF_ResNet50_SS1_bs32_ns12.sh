#!/bin/bash

python3 -B /data/users/amerinov/projects/holoassist/eval_action.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --raw_annotation_file /data/users/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/users/amerinov/data/holoassist/data-splits-v1_2 \
    --fga_map_file /data/users/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --num_classes 1887 \
    --base_model ResNet50 \
    --fusion_mode GSF \
    --pretrained SS1 \
    --batch_size 32 \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_segments 12 \
    --repetitions 10 \
    --checkpoint /data/users/amerinov/projects/holoassist/checkpoints/ResNet50_GSF_SS1_action_bs32_ns12/ResNet50_GSF_SS1_action_bs32_ns12_ep11.pth \
    | tee -a /data/users/amerinov/projects/holoassist/logs/eval_GSF_ResNet50_SS1_bs32_ns12.log
