import gc
import os
import time
from datetime import datetime
import argparse
import json

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.nn.functional as F

from src.opts.opts import parser
from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDatasetTest, prepare_clips_data_test
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from src.utils.meters import AverageMeter
from src.utils.metrics import calc_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--holoassist_dir", type=str, default="/data/users/amerinov/data/holoassist/HoloAssist")
    parser.add_argument("--test_action_clips_file", type=str, default="/data/users/amerinov/data/holoassist/test_action_clips.txt")
    parser.add_argument("--fga_map_file", type=str, default="/data/users/amerinov/data/holoassist/fine_grained_actions_map.txt")
    parser.add_argument("--base_model", type=str, default="InceptionV3")
    parser.add_argument("--fusion_mode", type=str, default="GSF")
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3, help="Number of spatial and temporal sampling to achieve better precision in evaluation.")
    parser.add_argument("--num_classes", type=int, default=1887)
    parser.add_argument("--checkpoint", type=str, default="/data/users/amerinov/projects/holoassist/checkpoints/holoassist_InceptionV3_GSF_action_11.pth", help="Best model weigths.")
    args = parser.parse_args()
    print(args)

    # Check folders.
    os.makedirs("logs", exist_ok=True)
    
    # Reproducibility.
    # Set up initial random states.
    make_reproducible(random_seed=0)

    #  ========================= DEFINE MODEL =========================
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoModel(
        num_classes=args.num_classes, 
        num_segments=args.num_segments, 
        base_model=args.base_model,
        fusion_mode=args.fusion_mode,
        verbose=False,
    ).to(device)

    input_size = model.input_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    div = model.div
    
    # Parallel!
    # model = torch.nn.DataParallel(model).to(device)

    #  ========================= LOAD MODEL STATE =========================
    # 

    checkpoint = torch.load(f=args.checkpoint, map_location=device)
    model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=True)

    #  ========================= PREPARE CLIPS DATA =========================
    #

    key_list, video_name_arr, start_arr, end_arr = prepare_clips_data_test(
        holoassist_dir=args.holoassist_dir,
        test_action_clips_file=args.test_action_clips_file,
    )

    #  ========================= REPETITIONS =========================
    # 

    # Make repetitions using outer loop to perform spatial and temporal sampling
    # to achieve better precision in evaluation. Inside each repeats we will
    # initialize dataset with new spatial and temporal sampling.

    # Let's average logits, since we use the same model.
    # However, for different models we will average (calibrated?) probabilities.

    logits = torch.empty((args.repetitions, len(video_name_arr), args.num_classes), dtype=torch.float32)

    for repeat_id in range(args.repetitions):

        print(
            f"\nrepeat_id={repeat_id}",
            f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
            flush=True
        )

        #  ========================= DATALOADER =========================
        # 

        # Make dataloader for each repeat.

        transform = Compose([
            GroupMultiScaleCrop(input_size=input_size, scales=[1, .875]),
            Stack(),
            ToTorchFormatTensor(div=div),
            GroupNormalize(mean=input_mean, std=input_std),
        ])
        dataset = VideoDatasetTest(
            holoassist_dir=args.holoassist_dir,
            video_name_arr=video_name_arr,
            start_arr=start_arr,
            end_arr=end_arr,
            num_segments=args.num_segments,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers, 
            drop_last=False, 
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
        )

        #  ========================= EVALUATION LOOP =========================
        # 

        processed_clips = 0
        
        model.eval()
        with torch.no_grad():
            for batch_id, xs in enumerate(dataloader):
            
                xs = xs.to(device) # video batch with image sequences [n, t_c, h, w]

                # Make prediction for validation batch
                batch_logits = model(xs) # [n, num_classes]
 
                logits[repeat_id, processed_clips:processed_clips + xs.size(0), :] = batch_logits
                processed_clips += xs.size(0)

                if batch_id % 100 == 0:
                    print(f"batch_id={batch_id:04d}/{len(dataloader):04d}",
                          f"processed_clips={processed_clips}/{len(dataset):04d}",
                          f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
                          flush=True)
    
    # Make average prediction for each clip
    logits = torch.mean(logits, dim=0, keepdim=False) # [num_clips, num_classes]
    top5_logits, top5_indices = torch.topk(logits, k=5, dim=1) # top5_indices: [num_clips, 5]
    
    # And create a dictionary with keys of format "{videoname}_{starttime}_{endtime}"
    top5 = {}
    top5["modality"] = "RGB"
    for k,v in zip(key_list, top5_indices):
        top5[k] = v.tolist()

    # Save dictionary
    with open(file="checkpoints/pred.json", mode='w') as f:
        json.dump(top5, f)

    print("Done")