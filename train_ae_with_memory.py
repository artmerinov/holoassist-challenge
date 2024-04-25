import os
import argparse
import time
import numpy as np
from datetime import datetime
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from src.utils.meters import AverageMeter
from src.utils.reproducibility import make_reproducible
from src.models.conv_ae_with_memory import ConvAEMemory
from src.dataset.video_dataset_ae import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--holoassist_dir', type=str, default="/data/amerinov/data/holoassist")
    parser.add_argument('--raw_annotation_file', type=str, default="/data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    parser.add_argument('--split_dir', type=str, default="/data/amerinov/data/holoassist/data-splits-v1")
    parser.add_argument('--fine_grained_actions_map_file', type=str, default="/data/amerinov/data/holoassist/fine_grained_actions_map.txt")
    parser.add_argument('--dataset_name', type=str, default="holoassist")
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clip_gradient", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--memory_size", type=int, default=32)
    parser.add_argument("--memory_dim", type=int, default=512)
    args = parser.parse_args()
    print(args)

    # Reproducibility.
    # Set up initial random states.
    make_reproducible(random_seed=0)
    
    clip_path_to_video_arr, clip_start_arr, clip_end_arr, clip_is_anomaly_arr = prepare_clips_data(
        raw_annotation_file=args.raw_annotation_file,
        holoassist_dir=args.holoassist_dir, 
        split_dir=args.split_dir,
        fine_grained_actions_map_file=args.fine_grained_actions_map_file,
        mode="train",
        anomaly_mode="normal",
    )

    transform = Compose([
        GroupMultiScaleCrop(input_size=224, scales=[1, .875]),
        Stack(),
        ToTorchFormatTensor(div=True),
        GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VideoDataset(
        clip_path_to_video_arr=clip_path_to_video_arr,
        clip_start_arr=clip_start_arr,
        clip_end_arr=clip_end_arr,
        clip_is_anomaly_arr=clip_is_anomaly_arr,
        num_segments=args.num_segments,
        transform=transform,
        mode="train"
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        drop_last=True, 
        pin_memory=False
    )

    # =========================== MODEL ====================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAEMemory(input_channels=args.num_segments * 3).to(device)
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # =========================== TRAIN LOOP ====================================

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=1e-3
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=args.num_epochs, 
        eta_min=1e-7, 
        last_epoch=-1
    )

    memory = F.normalize(torch.rand((args.memory_size, args.memory_dim), dtype=torch.float), dim=1)
    memory = memory.to(device)

    for epoch in range(args.num_epochs):
        
        # Reproducibility.
        # Set up random seed to current epoch
        # This is important to preserve reproducibility 
        # in case when we load model checkpoint.
        make_reproducible(random_seed=epoch)

        print(f"\nEpoch {epoch}", 
              f"LR={optimizer.param_groups[0]['lr']:.7f}",
              f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
              flush=True)
        
        epoch_rec_loss = AverageMeter()
        epoch_cmp_loss = AverageMeter()
        epoch_sep_loss = AverageMeter()
        epoch_ttl_loss = AverageMeter()

        for batch_id, (x, _) in enumerate(dataloader):
        
            x = x.to(device)
            x_hat, query, augmented_query, memory, score_query, score_memory, sep_loss, cmp_loss = model(x, memory, train=True)

            # Calculate total loss
            rec_loss = torch.mean((x - x_hat)**2) # by pixel
            cmp_loss = torch.mean(cmp_loss) # DataParallel makes a vector
            sep_loss = torch.mean(sep_loss) # DataParallel makes a vector
            ttl_loss = rec_loss + 0.1 * cmp_loss + 0.1 * sep_loss

            # Update epoch statistics
            epoch_rec_loss.update(value=rec_loss.item(), n=args.batch_size)
            epoch_cmp_loss.update(value=cmp_loss.item(), n=args.batch_size)
            epoch_sep_loss.update(value=sep_loss.item(), n=args.batch_size)
            epoch_ttl_loss.update(value=ttl_loss.item(), n=args.batch_size)

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Compute gradient of the loss wrt all learnable parameters
            ttl_loss.backward()

            # Clip computed gradients
            if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip_gradient)
            
            # Update the weights using optimizer
            optimizer.step()
            
            if batch_id % 20 == 0:
                print(
                    f"batch_id={batch_id:04d}/{len(dataloader):04d}",
                    f"|",
                    f"batch_rec_loss={rec_loss.item():.4f}", 
                    f"batch_cmp_loss={cmp_loss.item():.4f}",
                    f"batch_sep_loss={sep_loss.item():.4f}",
                    f"batch_ttl_loss={ttl_loss.item():.4f}",
                    f"|",
                    f"epoch_rec_loss={epoch_rec_loss.avg:.4f}",
                    f"epoch_cmp_loss={epoch_cmp_loss.avg:.4f}",
                    f"epoch_sep_loss={epoch_sep_loss.avg:.4f}",
                    f"epoch_ttl_loss={epoch_ttl_loss.avg:.4f}",
                    flush=True)

        print()

        # Adjust learning rate after training epoch
        lr_scheduler.step()

        # Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_fn = f"AE_{epoch:02d}.pth"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(obj=checkpoint, f=f"checkpoints/{checkpoint_fn}")
        print(f"Write model checkpoint {checkpoint_fn}", flush=True)

        # save memory items
        with open(f'checkpoints/memory_{epoch:02d}.npy', 'wb') as f:
            np.save(f, memory.detach().cpu().numpy())
