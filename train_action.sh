#!/bin/bash

#
# ------------------------------------------ InceptionV3 + GSF --------------------------------------------------------
#

python3 -B /data/users/amerinov/projects/holoassist/train_action.py \
    --holoassist_dir /data/users/amerinov/data/holoassist \
    --raw_annotation_file /data/users/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
    --split_dir /data/users/amerinov/data/holoassist/data-splits-v1 \
    --fga_map_file /data/users/amerinov/data/holoassist/fine_grained_actions_map.txt \
    --num_classes 1887 \
    --base_model InceptionV3 \
    --fusion_mode GSF \
    --num_epochs 1 \
    --batch_size 32 \
    --num_workers 8 \
    --num_segments 8 \
    --lr 0.01 \
    --clip_gradient 20 \
    | tee -a /data/users/amerinov/projects/holoassist/logs/train_action_inceptionv3_gsf.log

#
# ------------------------------------------ ResNet101 + GSF --------------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fga_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 1887 \
#     --base_model ResNet101 \
#     --fusion_mode GSF \
#     --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_ResNet101_GSF_action_00.pth \
#     --num_epochs 15 \
#     --batch_size 16 \
#     --num_workers 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_resnet101_gsf.log &

#
# ------------------------------------------ ResNet50 + GSF --------------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fga_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 1887 \
#     --base_model ResNet50 \
#     --fusion_mode GSF \
#     --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_ResNet50_GSF_action_02.pth \
#     --num_epochs 15 \
#     --batch_size 16 \
#     --num_workers 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_resnet50_gsf.log &

#
# ---------------------------------------------------- TimeSformer ----------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fga_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 1887 \
#     --base_model TimeSformer \
#     --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_TimeSformer_None_action_01.pth \
#     --num_epochs 20 \
#     --batch_size 16 \
#     --num_workers 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_timesformer.log &

#
# ---------------------------------------------------- HORST ----------------------------------------------------------
#

# nohup python3 -B train_action.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fga_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 1887 \
#     --base_model HORST \
#     --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_HORST_None_action_13.pth \
#     --num_epochs 20 \
#     --batch_size 16 \
#     --num_workers 4 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_action_horst.log &