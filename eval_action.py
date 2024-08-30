import os
import argparse
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from src.utils.metrics import calc_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--holoassist_dir", type=str)
    parser.add_argument("--raw_annotation_file", type=str)
    parser.add_argument("--split_dir", type=str)
    parser.add_argument("--fga_map_file", type=str)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--fusion_mode", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--num_segments", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--repetitions", type=int, help="Number of spatial and temporal sampling to achieve better precision in evaluation.")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--checkpoint", type=str, help="Best model weigths.")
    args = parser.parse_args()
    print(args)

    # Check folders.
    os.makedirs("logs", exist_ok=True)

    # Reproducibility.
    # Set up initial random states.
    make_reproducible(random_seed=0)
    
    #  ========================= LOAD MODEL =========================
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoModel(
        num_classes=args.num_classes, 
        num_segments=args.num_segments, 
        base_model=args.base_model,
        fusion_mode=args.fusion_mode,
        verbose=False,
        pretrained=args.pretrained,
        mode="validation",
    ).to(device)

    input_size = model.input_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    div = model.div

    checkpoint = torch.load(f=args.checkpoint, map_location=device)
    model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=True)

    #  ========================= PREPARE CLIPS DATA =========================
    # 

    video_name_arr, start_arr, end_arr, label_arr = prepare_clips_data(
        raw_annotation_file=args.raw_annotation_file,
        holoassist_dir=args.holoassist_dir, 
        split_dir=args.split_dir,
        fga_map_file=args.fga_map_file,
        mode="validation",
        task="action",
        debug=False,
    )

    #  ========================= REPETITIONS =========================
    # 

    # Make repetitions using outer loop to perform spatial and temporal sampling
    # to achieve better precision in evaluation. Inside each repeats we will
    # initialize dataset with new spatial and temporal sampling.

    # Let's average logits, since we use the same model.
    # However, for different models we will average (calibrated?) probabilities.

    logits = torch.empty((args.repetitions, len(video_name_arr), args.num_classes), dtype=torch.float32)
    trues = torch.empty(len(video_name_arr), dtype=torch.int32)

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
        dataset = VideoDataset(
            holoassist_dir=args.holoassist_dir,
            video_name_arr=video_name_arr,
            start_arr=start_arr,
            end_arr=end_arr,
            label_arr=label_arr,
            num_segments=args.num_segments,
            transform=transform,
            mode="validation",
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
            for batch_id, batch in enumerate(dataloader):
            
                xs = batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
                ys = batch[1].to(device) # video batch labels [n]

                # Make prediction for the batch
                batch_logits = model(xs) # [n, num_classes]
                
                logits[repeat_id, processed_clips:processed_clips + xs.size(0), :] = batch_logits
                trues[processed_clips:processed_clips + xs.size(0)] = ys
                processed_clips += xs.size(0)

                if batch_id % 100 == 0:

                    acc1, acc5  = calc_accuracy(
                        preds=torch.mean(logits[:, :processed_clips + xs.size(0), :], dim=0), 
                        labels=trues[:processed_clips + xs.size(0)], 
                        topk=(1,5)
                    )
                    print(f"batch_id={batch_id:04d}/{len(dataloader):04d}",
                          f"processed_clips={processed_clips}/{len(dataset):04d}",
                          f"acc@1={acc1:.3f}",
                          f"acc@5={acc5:.3f}",
                          f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
                          flush=True)
    
        # flops, macs, params = calculate_flops(
        #     model=model, 
        #     input_shape=tuple(xs.size()),
        #     output_as_string=True,
        #     output_precision=4
        # )
        # print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
        
        acc1, acc5  = calc_accuracy(
            preds=torch.mean(logits, dim=0), 
            labels=trues, 
            topk=(1,5)
        )
        print()
        print(f"Repeat {repeat_id} statistics", flush=True)
        print(f"repeat_id={repeat_id}", 
              f"acc@1={acc1:.3f}",
              f"acc@5={acc5:.3f}")
        print()
        
    print("DONE.")