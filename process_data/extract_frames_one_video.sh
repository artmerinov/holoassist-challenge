#!/bin/bash

python3 -B extract_frames_one_video.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --video_name R071-19July-BigPrinter \
    --fps 10 \
    --width 640 \
    --height 350 \
    --threads 16 \
