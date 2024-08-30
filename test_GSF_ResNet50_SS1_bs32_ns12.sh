#!/bin/bash

python3 -B /data/users/amerinov/projects/holoassist/test_action.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --fga_map_file /data/users/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --test_action_clips_file /data/users/amerinov/data/holoassist/test_action_clips-v1_2.txt \
    --num_classes 1887 \
    --base_model ResNet50 \
    --fusion_mode GSF \
    --pretrained SS1 \
    --batch_size 32 \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_segments 12 \
    --repetitions 10 \
    --checkpoint /data/users/amerinov/projects/holoassist/checkpoints/GSF_ResNet50_SS1_bs32_ns12/GSF_ResNet50_SS1_bs32_ns12_ep15.pth \
    --checkpoint_folder /data/users/amerinov/projects/holoassist/checkpoints/GSF_ResNet50_SS1_bs32_ns12 \
    