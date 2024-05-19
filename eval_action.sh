#!/bin/bash

#
# ------------------------------------------ InceptionV3 + GSF --------------------------------------------------------
#

python3 -B /data/users/amerinov/projects/holoassist/eval_action.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --raw_annotation_file /data/users/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/users/amerinov/data/holoassist/data-splits-v1 \
    --fga_map_file /data/users/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --num_classes 1887 \
    --base_model InceptionV3 \
    --fusion_mode GSF \
    --batch_size 32 \
    --num_workers 12 \
    --prefetch_factor 4 \
    --num_segments 8 \
    --repetitions 3 \
    --checkpoint /data/users/amerinov/projects/holoassist/checkpoints/holoassist_InceptionV3_GSF_action_11.pth \
    | tee -a /data/users/amerinov/projects/holoassist/logs/eval_action_inceptionv3_gsf.log
