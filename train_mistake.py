import os
import time
from datetime import datetime
from sklearn.metrics import brier_score_loss, roc_auc_score, classification_report

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from src.opts.opts import parser
from src.utils.reproducibility import make_reproducible
from src.utils.meters import AverageMeter
from src.utils.metrics import find_best_threshold, calc_metrics_by_threshold, save_vis_pr_curve
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from src.models.model import VideoModel


if __name__ == "__main__":

    # Check folders
    os.makedirs("pics", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Reproducibility.
    # Set up initial random states.
    make_reproducible(random_seed=0)

    # Load config.
    args = parser.parse_args()
    print(args)

    # TODO: move num_classes to config
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
    div = model.div
    learnable_named_parameters = model.learnable_named_parameters
    
    # Parallel!
    model = torch.nn.DataParallel(model).to(device)

    # Load weigths from pretrained model using action recognition task
    # TODO: move pretrained model to config
    checkpoint = torch.load(f='checkpoints/holoassist_InceptionV3_GSF_action_10.pth', map_location=torch.device(device))
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
        ToTorchFormatTensor(div=div),
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
        ToTorchFormatTensor(div=div),
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

    # TODO: try weighted loss function
    # e.g. weight = torch.tensor([6/100, (100-6)/100]), 
    # so 0.06 for class 0 (no mistake) and 0.96 for class 1 (mistake), 
    # since mistake is more important
    # or weigth = torch.tensor([6/6, 6/94])
    # torch.tensor([6277/118482, (118482-6277)/118482]),

    criterion = nn.CrossEntropyLoss(
        weight=None
    ).to(device)
    
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

        tr_epoch_trues = torch.empty(len(tr_dataset), dtype=torch.int32)
        tr_epoch_probs = torch.empty((len(tr_dataset), 2), dtype=torch.float32)
        
        tr_clips_processed_total = 0
        tr_clips_processed_mistake = 0
        tr_clips_processed_correct = 0

        model.train()
        for tr_batch_id, tr_batch in enumerate(tr_dataloader):
            
            tr_x = tr_batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
            tr_y = tr_batch[1].to(device) # video batch labels [n]

            # Make predictions for train batch
            tr_logits = model(tr_x)
            tr_loss = criterion(tr_logits, tr_y)

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
            tr_epoch_loss.update(value=tr_loss.item(), n=tr_x.size(0))

            # Store predictions
            tr_epoch_probs[tr_clips_processed_total: tr_clips_processed_total + len(tr_x)] = F.softmax(tr_logits, dim=1).detach()
            tr_epoch_trues[tr_clips_processed_total: tr_clips_processed_total + len(tr_x)] = tr_y.detach()
            
            # Keep track on total/correct/mistake clips count
            tr_clips_processed_total += len(tr_x)
            tr_clips_processed_mistake += (tr_y.detach() == 1).sum()
            tr_clips_processed_correct += (tr_y.detach() == 0).sum()

            if tr_batch_id % 20 == 0:

                if tr_epoch_trues[:tr_clips_processed_total].sum() > 0:

                    # Calculate running statistics
                    # Note, that more recent predictions
                    # are done with updated model, therefore, let's
                    # use only 10000 recent predictions
                    
                    # TODO: rework window -> use exp moving average instead

                    window_size = 10000
                    window_st = max(0, tr_clips_processed_total - window_size)
                    window_en = tr_clips_processed_total

                    # Probability-based
                    tr_rocauc = roc_auc_score(
                        y_true=tr_epoch_trues[window_st:window_en], 
                        y_score=tr_epoch_probs[window_st:window_en, 1],
                        average="macro"
                    )

                    # Threshold-based
                    tr_thr = find_best_threshold(
                        trues=tr_epoch_trues[window_st:window_en],
                        probs=tr_epoch_probs[window_st:window_en],
                    )
                    tr_precision, tr_recall, tr_f1score = calc_metrics_by_threshold(
                        thr=tr_thr,
                        trues=tr_epoch_trues[window_st:window_en],
                        probs=tr_epoch_probs[window_st:window_en],
                    )

                else:
                    # If there is only one class
                    tr_rocauc = 0.0
                    tr_thr = 0.0
                    tr_precision = 0.0
                    tr_recall = 0.0
                    tr_f1score = 0.0

                print(f"batch_id={tr_batch_id:04d}/{len(tr_dataloader):04d}",
                      f"total={tr_clips_processed_total:06d}",
                      f"mistake={tr_clips_processed_mistake:06d}",
                      f"correct={tr_clips_processed_correct:06d}",
                      f"|",
                      f"epoch_loss={tr_epoch_loss.avg:.3f}",
                      f"window_rocauc={tr_rocauc:.3f}",
                      f"|",
                      f"thr={tr_thr:.3f}",
                      f"window_precision={tr_precision:.3f}",
                      f"window_recall={tr_recall:.3f}",
                      f"window_f1score={tr_f1score:.3f}",
                      flush=True)
                
            del tr_logits, tr_loss

        print(classification_report(
            y_true=tr_epoch_trues,
            y_pred=(tr_epoch_probs[:,1] >= tr_thr).int(), 
            zero_division=True,
            digits=3
        ))

        # Adjust learning rate after training epoch
        lr_scheduler.step()

        # ================================== VALIDATION ==================================
        # 

        print(f"\nVALIDATION")

        va_epoch_loss = AverageMeter()
        va_epoch_trues = torch.empty(len(va_dataset), dtype=torch.int32)
        va_epoch_probs = torch.empty((len(va_dataset), 2), dtype=torch.float32)
        
        va_clips_processed_total = 0
        va_clips_processed_mistake = 0
        va_clips_processed_correct = 0

        model.eval()
        with torch.no_grad():
            for va_batch_id, va_batch in enumerate(va_dataloader):
                
                va_x = va_batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
                va_y = va_batch[1].to(device) # video batch labels [n]

                # Make prediction for validation batch
                va_logits = model(va_x)
                va_loss = criterion(va_logits, va_y)

                # Keep track of epoch metrics
                va_epoch_loss.update(value=va_loss.item(), n=va_x.size(0))

                # Store predictions
                va_epoch_probs[va_clips_processed_total: va_clips_processed_total + len(va_x)] = F.softmax(va_logits, dim=1).detach()
                va_epoch_trues[va_clips_processed_total: va_clips_processed_total + len(va_x)] = va_y.detach()
                
                # Keep traack on total/correct/mistake clips count
                va_clips_processed_total += len(va_x)
                va_clips_processed_mistake += (va_y.detach() == 1).sum()
                va_clips_processed_correct += (va_y.detach() == 0).sum()
                
                if va_batch_id % 10 == 0:

                    # Calculate running statistics
                    # Note, that the model is fixed here, therefore
                    # we will use all predictions till thee current batch
                    # to output statistics

                    if va_epoch_trues[:va_clips_processed_total].sum() > 0:
                        
                        # Probability-based
                        va_rocauc = roc_auc_score(
                            y_true=va_epoch_trues[:va_clips_processed_total], 
                            y_score=va_epoch_probs[:va_clips_processed_total, 1],
                            average="macro"
                        )

                        # Threshold-based
                        va_thr = find_best_threshold(
                            trues=va_epoch_trues[:va_clips_processed_total],
                            probs=va_epoch_probs[:va_clips_processed_total],
                        )
                        va_precision, va_recall, va_f1score = calc_metrics_by_threshold(
                            thr=va_thr,
                            trues=va_epoch_trues[:va_clips_processed_total],
                            probs=va_epoch_probs[:va_clips_processed_total], 
                        )

                    else:
                        # If there is only one class
                        va_rocauc = 0.0
                        va_thr = 0.0
                        va_precision = 0.0
                        va_recall = 0.0
                        va_f1score = 0.0

                    print(f"batch_id={va_batch_id:04d}/{len(va_dataloader):04d}",
                          f"total={va_clips_processed_total:06d}",
                          f"mistake={va_clips_processed_mistake:06d}",
                          f"correct={va_clips_processed_correct:06d}",
                          f"|",
                          f"epoch_loss={va_epoch_loss.avg:.3f}",
                          f"rocauc={va_rocauc:.3f}",
                          f"|",
                          f"thr={va_thr:.3f}",
                          f"precision={va_precision:.3f}",
                          f"recall={va_recall:.3f}",
                          f"f1score={va_f1score:.3f}",
                          flush=True)
                    
                del va_logits, va_loss

        print(classification_report(
            y_true=va_epoch_trues,
            y_pred=(va_epoch_probs[:,1] >= va_thr).int(), 
            zero_division=True,
            digits=3
        ))
        
        # Save model checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(
            obj=checkpoint, 
            f=f"checkpoints/{args.dataset_name}_{args.base_model}_{args.fusion_mode}_mistake_{epoch:02d}.pth"
        )

        # Save pics
        save_vis_pr_curve(
            trues=va_epoch_trues,
            probs=va_epoch_probs,
            fname=f"pics/va_PR_{epoch:02d}.png"
        )
        save_vis_pr_curve(
            trues=tr_epoch_trues,
            probs=tr_epoch_probs,
            fname=f"pics/tr_PR_{epoch:02d}.png"
        )