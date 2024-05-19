#!/bin/bash

python3 -B extract_frames.py \
    --holoassist_dir /data/users/amerinov/data/holoassist/HoloAssist \
    --fps 10 \
    --width 640 \
    --height 350 \
    --threads 16 \
    | tee -a /data/users/amerinov/projects/holoassist/logs/extract_frames.log
