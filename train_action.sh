#!/bin/bash

#
# ------------------------------------------ InceptionV3 + GSF --------------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 1887 \
#     --base_model InceptionV3 \
#     --fusion_mode GSF \
#     --num_epochs 10 \
#     --batch_size 16 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_inceptionv3_gsf.log &

#
# ---------------------------------------------------- TimeSformer ----------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 1887 \
#     --base_model TimeSformer \
#     --num_epochs 20 \
#     --batch_size 16 \
#     --prefetch_factor 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_timesformer.log &

#
# ---------------------------------------------------- HORST ----------------------------------------------------------
#

nohup python3 -B train_action.py \
    --holoassist_dir /data/amerinov/data/holoassist \
    --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
    --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --num_classes 1887 \
    --base_model HORST \
    --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_HORST_None_action_13.pth \
    --num_epochs 30 \
    --batch_size 16 \
    --prefetch_factor 4 \
    --num_segments 8 \
    --lr 0.01 \
    --clip_gradient 20 \
    | tee -a logs/train_action_horst.log &