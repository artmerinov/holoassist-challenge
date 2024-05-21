import gc
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.opts.opts import parser
from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from src.utils.meters import AverageMeter
from src.utils.metrics import calc_accuracy


if __name__ == "__main__":

    # Check folders.
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Reproducibility.
    # Set up initial random states.
    make_reproducible(random_seed=0)

    # Load config.
    args = parser.parse_args()
    print(args)

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
    learnable_named_parameters = model.learnable_named_parameters
    
    # Parallel!
    # if args.base_model != "HORST":
    #     model = torch.nn.DataParallel(model).to(device)

    #  ========================= TRAIN DATA =========================
    # 

    print("tr_dataset", flush=True)

    tr_video_name_arr, tr_start_arr, tr_end_arr, tr_label_arr = prepare_clips_data(
        raw_annotation_file=args.raw_annotation_file,
        holoassist_dir=args.holoassist_dir, 
        split_dir=args.split_dir,
        fga_map_file=args.fga_map_file,
        mode="train",
        task="action",
        debug=args.debug,
    )
    tr_transform = Compose([
        GroupMultiScaleCrop(input_size=input_size, scales=[1, .875]),
        Stack(),
        ToTorchFormatTensor(div=div),
        GroupNormalize(mean=input_mean, std=input_std),
    ])
    tr_dataset = VideoDataset(
        holoassist_dir=args.holoassist_dir,
        video_name_arr=tr_video_name_arr,
        start_arr=tr_start_arr,
        end_arr=tr_end_arr,
        label_arr=tr_label_arr,
        num_segments=args.num_segments,
        transform=tr_transform,
        mode="train",
    )
    tr_dataloader = DataLoader(
        dataset=tr_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        drop_last=True,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    #  ========================= VALIDATION DATA =========================
    # 

    print("va_dataset", flush=True)

    va_video_name_arr, va_start_arr, va_end_arr, va_label_arr = prepare_clips_data(
        raw_annotation_file=args.raw_annotation_file,
        holoassist_dir=args.holoassist_dir, 
        split_dir=args.split_dir,
        fga_map_file=args.fga_map_file,
        mode="validation",
        task="action",
        debug=args.debug,
    )
    va_transform = Compose([
        GroupMultiScaleCrop(input_size=crop_size, scales=[1, .875]),
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
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # =====================================================================

    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.base_model in ["InceptionV3"]:
        lr_scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=[5, 10],
            gamma=0.5
        )
    else:
        raise ValueError()

    if args.resume:
        if os.path.isfile(args.resume):

            # Load checkpoint file that contains all the states
            print(f"=> Loading checkpoint {args.resume}")
            
            # Load state into model from checkpoint
            checkpoint = torch.load(f=args.resume)
            model.load_state_dict(state_dict=checkpoint['model_state_dict'])

            args.start_epoch = checkpoint['epoch'] + 1
            for i in range(args.start_epoch):
                optimizer.step()
                lr_scheduler.step()
        else:
            raise ValueError(f"=> No checkpoint found at {args.resume}")
        
    # ==================== Main train-validation loop =================================

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

        # TRAIN
        # =====
        print(f"\nTRAIN")

        tr_epoch_loss = AverageMeter()
        tr_epoch_acc1 = AverageMeter()
        tr_epoch_acc5 = AverageMeter()

        model.train()
        for tr_batch_id, tr_batch in enumerate(tr_dataloader):
            
            tr_x = tr_batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
            tr_y = tr_batch[1].to(device) # video batch labels [n]

            # print("Device of tr_x:", tr_x.device)
            # print("Shape of tr_x:", tr_x.shape)
            # print("Data type of tr_x:", tr_x.dtype)

            # print("Device of tr_y:", tr_y.device)
            # print("Shape of tr_y:", tr_y.shape)
            # print("Data type of tr_y:", tr_y.dtype)

            # Perform forward pass
            tr_preds = model(tr_x)

            # print("Device of tr_preds:", tr_preds.device)
            # print("Shape of tr_preds:", tr_preds.shape)
            # print("Data type of tr_preds:", tr_preds.dtype)
            
            tr_loss = criterion(tr_preds, tr_y)

            # print(tr_loss)

            # Set the gradients to None after calling backward(), rather than set to zero. 
            # This can save memory, especially when dealing with large models or long sequences, 
            # where the gradients are sparse.
            optimizer.zero_grad(set_to_none=True)

            # Compute gradient of the loss wrt all learnable parameters
            tr_loss.backward()
            
            # Clip computed gradients
            if args.clip_gradient is not None:
                # Save the grad norm over all gradients together for debugging
                # purposes (to set up reasonable max value). This variable stores 
                # grad norm before clipping, however, gradients are modified in-place.
                grad_norm = clip_grad_norm_(
                    parameters=model.parameters(), 
                    max_norm=args.clip_gradient,
                    norm_type=2
                )
            
            # Update the weights using optimizer
            optimizer.step()

            # Calculate batch metrics
            tr_acc1, tr_acc5  = calc_accuracy(preds=tr_preds, labels=tr_y, topk=(1,5))

            # Keep track of epoch metrics (update for each batch)
            tr_epoch_loss.update(value=tr_loss.detach().item(), n=tr_x.size(0))
            tr_epoch_acc1.update(value=tr_acc1, n=tr_x.size(0))
            tr_epoch_acc5.update(value=tr_acc5, n=tr_x.size(0))

            if tr_batch_id % 20 == 0:

                print(f"tr_batch_id={tr_batch_id:04d}/{len(tr_dataloader):04d}",
                      f"|",
                      f"tr_batch_loss={tr_loss.detach().item():.3f}",
                      f"tr_batch_acc@1={tr_acc1:.3f}",
                      f"tr_batch_acc@5={tr_acc5:.3f}",
                      f"|",
                      f"tr_epoch_loss={tr_epoch_loss.avg:.3f}",
                      f"tr_epoch_acc@1={tr_epoch_acc1.avg:.3f}",
                      f"tr_epoch_acc@5={tr_epoch_acc5.avg:.3f}",
                      f"|",
                      f"grad_norm={grad_norm:.3f}",
                      f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
                      flush=True)
                
            del tr_preds, tr_loss, tr_acc1, tr_acc5, tr_batch, tr_x, tr_y
            gc.collect()
        
        # Adjust learning rate after training epoch
        lr_scheduler.step()

        # VALIDATION
        # ==========
        print(f"\nVALIDATION")

        va_epoch_loss = AverageMeter()
        va_epoch_acc1 = AverageMeter()
        va_epoch_acc5 = AverageMeter()

        model.eval()
        with torch.no_grad():
            for va_batch_id, va_batch in enumerate(va_dataloader):
                
                va_x = va_batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
                va_y = va_batch[1].to(device) # video batch labels [n]

                # Make prediction for validation batch
                va_preds = model(va_x)
                va_loss = criterion(va_preds, va_y)
                va_acc1, va_acc5 = calc_accuracy(va_preds, va_y, topk=(1,5))

                # Keep track of epoch metrics
                va_epoch_loss.update(value=va_loss.detach().item(), n=va_x.size(0))
                va_epoch_acc1.update(value=va_acc1, n=va_x.size(0))
                va_epoch_acc5.update(value=va_acc5, n=va_x.size(0))
                
                if va_batch_id % 10 == 0:
                    print(f"va_batch_id={va_batch_id:04d}/{len(va_dataloader):04d}",
                          f"|",
                          f"va_batch_loss={va_loss.detach().item():.3f}",
                          f"va_batch_acc@1={va_acc1:.3f}",
                          f"va_batch_acc@5={va_acc5:.3f}",
                          f"|",
                          f"va_epoch_loss={va_epoch_loss.avg:.3f}",
                          f"va_epoch_acc@1={va_epoch_acc1.avg:.3f}",
                          f"va_epoch_acc@5={va_epoch_acc5.avg:.3f}",
                          f"|",
                          f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
                          flush=True)
                
                del va_preds, va_loss, va_acc1, va_acc5, va_batch, va_x, va_y
                gc.collect()

        # ----------------------------------------------------------------------------------

        print()
        print(f"Epoch {epoch:02d} statistics", flush=True)
        print(f"tr_epoch_loss={tr_epoch_loss.avg:.3}",
              f"va_epoch_loss={va_epoch_loss.avg:.3}",
              f"|",
              f"tr_epoch_acc@1={tr_epoch_acc1.avg:.3}",
              f"tr_epoch_acc@5={tr_epoch_acc5.avg:.3}",
              f"|",
              f"va_epoch_acc@1={va_epoch_acc1.avg:.3}",
              f"va_epoch_acc@5={va_epoch_acc5.avg:.3}",
              f"\n", flush=True)
        
        # Save model checkpoint
        checkpoint_fn = f"holoassist_{args.base_model}_{args.fusion_mode}_action_{epoch:02d}.pth"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(obj=checkpoint, f=f"checkpoints/{checkpoint_fn}")
        print(f"Write model checkpoint {checkpoint_fn}", flush=True)

        del checkpoint
        gc.collect()