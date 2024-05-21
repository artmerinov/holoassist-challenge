#!/bin/bash

#
# ------------------------------------------ InceptionV3 + GSF --------------------------------------------------------
#

python3 -B /data/users/amerinov/projects/holoassist/test_action.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --fga_map_file /data/users/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --num_classes 1887 \
    --base_model InceptionV3 \
    --fusion_mode GSF \
    --batch_size 32 \
    --num_workers 28 \
    --prefetch_factor 8 \
    --num_segments 8 \
    --repetitions 2 \
    --test_action_clips_file /data/users/amerinov/data/holoassist/test_action_clips.txt \
    --checkpoint /data/users/amerinov/projects/holoassist/checkpoints/holoassist_InceptionV3_GSF_action_11.pth \
