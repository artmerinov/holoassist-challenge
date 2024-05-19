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
from src.utils.meters import AverageMeter
from src.utils.metrics import calc_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--holoassist_dir", type=str, default="/data/users/amerinov/data/holoassist/HoloAssist")
    parser.add_argument("--raw_annotation_file", type=str, default="/data/users/amerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    parser.add_argument("--split_dir", type=str, default="/data/users/amerinov/data/holoassist/data-splits-v1")
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

    va_video_name_arr, va_start_arr, va_end_arr, va_label_arr = prepare_clips_data(
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

    global_acc1 = AverageMeter()
    global_acc5 = AverageMeter()

    for repeat_id in range(args.repetitions):

        print(
            f"\nrepeat_id={repeat_id}",
            f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
            flush=True
        )

        #  ========================= DATALOADER =========================
        # 

        # Make dataloader for each repeat.

        va_transform = Compose([
            GroupMultiScaleCrop(input_size=input_size, scales=[1, .875]),
            Stack(),
            ToTorchFormatTensor(div=div),
            GroupNormalize(mean=input_mean, std=input_std),
        ])
        va_dataset = VideoDataset(
            holoassist_dir=args.holoassist_dir,
            video_name_arr=va_video_name_arr,
            start_arr=va_start_arr,
            end_arr=va_end_arr,
            label_arr=va_label_arr,
            num_segments=args.num_segments,
            transform=va_transform,
            mode="validation",
        )
        va_dataloader = DataLoader(
            dataset=va_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers, 
            drop_last=False, 
            pin_memory=False,
            prefetch_factor=args.prefetch_factor,
        )

        #  ========================= EVALUATION LOOP =========================
        # 
        
        model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(va_dataloader):
            
                batch_x = batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
                batch_y = batch[1].to(device) # video batch labels [n]

                # # check that repetitions work
                # if batch_id == 3:
                #     print(batch_x[0], flush=True)
                #     print(batch_y, flush=True)
                
                # Make prediction for validation batch
                batch_logits = model(batch_x)

                # Batch metrics
                batch_acc1, batch_acc5 = calc_accuracy(batch_logits, batch_y, topk=(1,5))

                # Update global metrics
                global_acc1.update(batch_acc1, n=batch_logits.size(0))
                global_acc5.update(batch_acc5, n=batch_logits.size(0))

                if batch_id % 10 == 0:
                    print(f"repeat_id={repeat_id}/{args.repetitions}",
                          f"batch_id={batch_id:04d}/{len(va_dataloader):04d}",
                          f"|",
                          f"batch_acc@1={batch_acc1:.3f}",
                          f"batch_acc@5={batch_acc5:.3f}",
                          f"|",
                          f"global_acc@1={global_acc1.avg:.3f}",
                          f"global_acc@5={global_acc5.avg:.3f}",
                          flush=True)
    
    print("DONE.")