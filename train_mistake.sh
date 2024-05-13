#!/bin/bash

#
# ------------------------------------------ InceptionV3 + GSF --------------------------------------------------------
#

# nohup python3 -B train_mistake.py \
#     --holoassist_dir /data/amerinov/data/holoassist \
#     --raw_annotation_file /data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json \
#     --split_dir /data/amerinov/data/holoassist/data-splits-v1 \
#     --fine_grained_actions_map_file /data/amerinov/data/holoassist/fine_grained_actions_map.txt \
#     --num_classes 2 \
#     --base_model InceptionV3 \
#     --fusion_mode GSF \
#     --resume /data/amerinov/projects/holoassist/checkpoints/holoassist_InceptionV3_GSF_mistake_03.pth \
#     --num_epochs 10 \
#     --batch_size 16 \
#     --num_workers 8 \
#     --num_segments 8 \
#     --lr 0.01 \
#     --clip_gradient 20 \
#     | tee -a logs/train_mistake_inceptionv3_gsf.log &