#!/bin/bash

#
# ------------------------------------------ InceptionV3 + GSF --------------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --dataset_name holoassist \
#     --base_model InceptionV3 \
#     --fusion_mode GSF \
#     --dropout 0.5 \
#     --num_epochs 10 \
#     --batch_size 16 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_inceptionv3_gsf.log &

# python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --dataset_name holoassist \
#     --base_model InceptionV3 \
#     --fusion_mode GSF \
#     --dropout 0.5 \
#     --num_epochs 10 \
#     --batch_size 16 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_inceptionv3_gsf.log

#
# ---------------------------------------------------- TimeSformer ----------------------------------------------------
#

nohup python3 -B train_action.py \
    --holoassist_dir /data/amerinov/data/holoassist \
    --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
    --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --dataset_name holoassist \
    --base_model TimeSformer \
    --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_TimeSformer_None_action_08.pth \
    --num_epochs 15 \
    --batch_size 16 \
    --prefetch_factor 4 \
    --num_segments 8 \
    --lr 0.01 \
    --clip_gradient 20 \
    | tee -a logs/train_action_timesformer.log &

# python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --dataset_name holoassist \
#     --base_model TimeSformer \
#     --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_TimeSformer_None_action_08.pth \
#     --num_epochs 15 \
#     --batch_size 16 \
#     --prefetch_factor 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_timesformer.log

#
# ---------------------------------------------------- HORST ----------------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --dataset_name holoassist \
#     --base_model HORST \
#     --dropout 0.5 \
#     --num_epochs 10 \
#     --batch_size 16 \
#     --prefetch_factor 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_horst.log &

# python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --dataset_name holoassist \
#     --base_model HORST \
#     --dropout 0.5 \
#     --num_epochs 10 \
#     --batch_size 16 \
#     --prefetch_factor 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_horst.log