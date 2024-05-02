import gc
import os
import time
import numpy as np
import random
from datetime import datetime
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from src.opts.opts import parser
from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import (
    GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize,
)
from src.utils.meters import AverageMeter


if __name__ == "__main__":

    # Reproducibility.
    # Set up initial random states.
    make_reproducible(random_seed=0)

    # Load config.
    args = parser.parse_args()
    print(args)

    if args.dataset_name == 'holoassist':
        num_classes = 2 # mistake or not
    else:
        raise NotImplementedError()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoModel(
        num_classes=num_classes, 
        num_segments=args.num_segments, 
        base_model=args.base_model,
        fusion_mode=args.fusion_mode,
        dropout=args.dropout,
        verbose=False,
    ).to(device)
    # print(model)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    learnable_named_parameters = model.learnable_named_parameters
    
    # Parallel!
    model = torch.nn.DataParallel(model).to(device)

    # Load weigths from pretrained model using action recognition task
    checkpoint = torch.load(f='checkpoints/holoassist_InceptionV3_GSF_10.pth', map_location=torch.device(device))
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'base_model.top_cls_fc' not in k}
    model.load_state_dict(state_dict=pretrained_dict, strict=False)

    #  ========================= TRAIN DATA =========================
    # 

    print("tr_dataset", flush=True)

    tr_clip_path_to_video_arr, tr_clip_start_arr, tr_clip_end_arr, _, tr_clip_mistake_arr = prepare_clips_data(
        raw_annotation_file=args.raw_annotation_file,
        holoassist_dir=args.holoassist_dir,
        split_dir=args.split_dir,
        fine_grained_actions_map_file=args.fine_grained_actions_map_file,
        mode="train",
    )
    tr_transform = Compose([
        GroupMultiScaleCrop(input_size=crop_size, scales=[1, .875]),
        Stack(),
        ToTorchFormatTensor(div=(args.base_model not in ['BNInception'])),
        GroupNormalize(mean=input_mean, std=input_std),
    ])

    tr_dataset = VideoDataset(
        clip_path_to_video_arr=tr_clip_path_to_video_arr,
        clip_start_arr=tr_clip_start_arr,
        clip_end_arr=tr_clip_end_arr,
        clip_label_arr=tr_clip_mistake_arr,
        num_segments=args.num_segments,
        transform=tr_transform,
        mode="train"
    )
    tr_dataloader = DataLoader(
        dataset=tr_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        drop_last=True,
        pin_memory=False,
    )

    #  ========================= VALIDATION DATA =========================
    # 

    print("va_dataset", flush=True)

    va_clip_path_to_video_arr, va_clip_start_arr, va_clip_end_arr, _, va_clip_mistake_arr = prepare_clips_data(
        raw_annotation_file=args.raw_annotation_file,
        holoassist_dir=args.holoassist_dir, 
        split_dir=args.split_dir,
        fine_grained_actions_map_file=args.fine_grained_actions_map_file,
        mode="validation",
    )
    va_transform = Compose([
        GroupMultiScaleCrop(input_size=crop_size, scales=[1, .875]),
        Stack(),
        ToTorchFormatTensor(div=(args.base_model not in ['BNInception'])),
        GroupNormalize(mean=input_mean, std=input_std),
    ])
    va_dataset = VideoDataset(
        clip_path_to_video_arr=va_clip_path_to_video_arr,
        clip_start_arr=va_clip_start_arr,
        clip_end_arr=va_clip_end_arr,
        clip_label_arr=va_clip_mistake_arr,
        num_segments=args.num_segments,
        transform=va_transform,
        mode="validation"
    )
    va_dataloader = DataLoader(
        dataset=va_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        drop_last=False, 
        pin_memory=False
    )

    # =====================================================================

    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=args.num_epochs, 
        eta_min=1e-7, 
        last_epoch=-1
    )

    if args.resume:
        if os.path.isfile(args.resume):

            # Load checkpoint file that contains all the states
            print(f"=> Loading checkpoint {args.resume}")
            checkpoint = torch.load(f=args.resume)

            # Load state from checkpoint
            model.load_state_dict(state_dict=checkpoint['model_state_dict'])
            optimizer.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(state_dict=checkpoint['lr_scheduler_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
        else:
            raise ValueError(f"=> No checkpoint found at {args.resume}")
        
    # ==================== Main train-validation loop =================================
    # 

    for epoch in range(args.start_epoch, args.num_epochs):

        # Reproducibility.
        # Set up random seed to current epoch
        # This is important to preserve reproducibility 
        # in case when we load model checkpoint.
        make_reproducible(random_seed=epoch)

        print(f"\nEpoch {epoch}", 
              f"LR={optimizer.param_groups[0]['lr']:.7f}",
              f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
              flush=True)

        # ================================== TRAIN ==================================
        # 

        print(f"\nTRAIN")

        tr_epoch_loss = AverageMeter()

        tr_epoch_trues = np.empty(len(tr_dataloader)*args.batch_size, dtype=np.int32)
        tr_epoch_preds = np.empty((len(tr_dataloader)*args.batch_size, 2), dtype=np.float32)
        tr_clips_processed = 0
        tr_clips_processed_mistake = 0
        tr_clips_processed_normal = 0

        model.train()
        for tr_batch_id, tr_batch in enumerate(tr_dataloader):
            
            tr_x = tr_batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
            tr_y = tr_batch[1].to(device) # video batch labels [n]

            # Make predictions for train batch
            tr_preds = model(tr_x)
            tr_loss = criterion(tr_preds, tr_y)

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Compute gradient of the loss wrt all learnable parameters
            tr_loss.backward()

            # Clip computed gradients
            if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip_gradient)
            
            # Update the weights using optimizer
            optimizer.step()

            # Keep track of epoch metrics (for each batch)
            tr_epoch_loss.update(value=tr_loss.item(), n=args.batch_size)

            # Store predictions
            tr_epoch_preds[tr_clips_processed: tr_clips_processed + len(tr_x)] = F.softmax(tr_preds, dim=1).cpu().detach().numpy()
            tr_epoch_trues[tr_clips_processed: tr_clips_processed + len(tr_x)] = tr_y.cpu().detach().numpy()
            tr_clips_processed += len(tr_x)
            tr_clips_processed_mistake += (tr_y.cpu().detach().numpy() == 1).sum()
            tr_clips_processed_normal += (tr_y.cpu().detach().numpy() == 0).sum()

            if tr_batch_id % 20 == 0:

                if tr_epoch_trues[:tr_clips_processed].sum() > 0:

                    # Calculate running statistics
                    # Note, that more recent predictions
                    # are done with updated model, therefore, let's
                    # use only 10000 recent predictions
                    
                    window_size = 10000
                    window_st = max(0, tr_clips_processed - window_size)
                    window_en = tr_clips_processed

                    tr_rocauc = roc_auc_score(
                        y_true=tr_epoch_trues[window_st:window_en], 
                        y_score=tr_epoch_preds[window_st:window_en, 1],
                        average="macro"
                    )
                    tr_f1 = f1_score(
                        y_true=tr_epoch_trues[window_st:window_en], 
                        y_pred=np.argmax(tr_epoch_preds[window_st:window_en], axis=1),
                        average='macro'
                    )
                else:
                    # If there is only one class
                    tr_rocauc = 0.0
                    tr_f1 = 0.0

                print(f"tr_batch_id={tr_batch_id:04d}/{len(tr_dataloader):04d}",
                      f"tr_clips={tr_clips_processed:06d}",
                      f"tr_clips_mistake={tr_clips_processed_mistake:06d}",
                      f"tr_clips_normal={tr_clips_processed_normal:06d}",
                      f"tr_batch_loss={tr_loss.item():.3f}",
                      f"tr_epoch_loss={tr_epoch_loss.avg:.3f}",
                      f"tr_window{window_size:06d}_rocauc={tr_rocauc:.3f}",
                      f"tr_window{window_size:06d}_f1={tr_f1:.3f}",
                      flush=True)
                
            del tr_preds, tr_loss
        
        # Adjust learning rate after training epoch
        lr_scheduler.step()


        # ================================== VALIDATION ==================================
        # 

        print(f"\nVALIDATION")

        va_epoch_loss = AverageMeter()
        va_epoch_trues = np.empty(len(va_dataloader)*args.batch_size, dtype=np.int32)
        va_epoch_preds = np.empty((len(va_dataloader)*args.batch_size, 2), dtype=np.float32)
        va_clips_processed = 0
        va_clips_processed_mistake = 0
        va_clips_processed_normal = 0

        model.eval()
        with torch.no_grad():
            for va_batch_id, va_batch in enumerate(va_dataloader):
                
                va_x = va_batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
                va_y = va_batch[1].to(device) # video batch labels [n]

                # Make prediction for validation batch
                va_preds = model(va_x)
                va_loss = criterion(va_preds, va_y)

                # Keep track of epoch metrics
                va_epoch_loss.update(value=va_loss.item(), n=args.batch_size)

                # Store predictions
                va_epoch_preds[va_clips_processed: va_clips_processed + len(va_x)] = F.softmax(va_preds, dim=1).cpu().detach().numpy()
                va_epoch_trues[va_clips_processed: va_clips_processed + len(va_x)] = va_y.cpu().detach().numpy()
                va_clips_processed += len(va_x)
                va_clips_processed_mistake += (va_y.cpu().detach().numpy() == 1).sum()
                va_clips_processed_normal += (va_y.cpu().detach().numpy() == 0).sum()
                
                if va_batch_id % 10 == 0:

                    # Calculate running statistics
                    # Note, that the model is fixed here, therefore
                    # we will use all predictions till thee current batch
                    # to output statistics

                    if va_epoch_trues[:va_clips_processed].sum() > 0:
                        
                        va_rocauc = roc_auc_score(
                            y_true=va_epoch_trues[:va_clips_processed], 
                            y_score=va_epoch_preds[:va_clips_processed, 1],
                            average="macro"
                        )
                        va_f1 = f1_score(
                            y_true=va_epoch_trues[:va_clips_processed], 
                            y_pred=np.argmax(va_epoch_preds[:va_clips_processed], axis=1), 
                            average='macro'
                        )
                    else:
                        # If there is only one class
                        va_rocauc = 0.0
                        va_f1 = 0.0

                    print(f"va_batch_id={va_batch_id:04d}/{len(va_dataloader):04d}",
                          f"va_clips={va_clips_processed:06d}",
                          f"va_clips_mistake={va_clips_processed_mistake:06d}",
                          f"va_clips_normal={va_clips_processed_normal:06d}",
                          f"va_batch_loss={va_loss.item():.3f}",
                          f"va_epoch_loss={va_epoch_loss.avg:.3f}",
                          f"va_window{va_clips_processed:06d}_rocauc={va_rocauc:.3f}",
                          f"va_window{va_clips_processed:06d}_f1={va_f1:.3f}",
                          flush=True)
                
                del va_preds, va_loss
        
        # Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_fn = f"{args.dataset_name}_{args.base_model}_{args.fusion_mode}_mistake_{epoch:02d}.pth"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(obj=checkpoint, f=f"checkpoints/{checkpoint_fn}")
        print(f"Write model checkpoint {checkpoint_fn}", flush=True)
