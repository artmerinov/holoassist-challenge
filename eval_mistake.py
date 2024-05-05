import argparse
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, confusion_matrix
from typing import List

import torch
from torchvision.transforms import Compose
import torch.nn.functional as F

from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from src.utils.metrics import find_best_f1score, calc_metrics_by_threshold


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # local run
    parser.add_argument("--holoassist_dir", type=str, default="/Users/artemmerinov/data/holoassist")
    parser.add_argument("--raw_annotation_file", type=str, default="/Users/artemmerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    parser.add_argument("--split_dir", type=str, default="/Users/artemmerinov/data/holoassist/data-splits-v1")
    parser.add_argument("--fine_grained_actions_map_file", type=str, default="/Users/artemmerinov/data/holoassist/fine_grained_actions_map.txt")

    # server run
    # parser.add_argument("--holoassist_dir", type=str, default="/data/amerinov/data/holoassist")
    # parser.add_argument("--raw_annotation_file", type=str, default="/data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    # parser.add_argument("--split_dir", type=str, default="/data/amerinov/data/holoassist/data-splits-v1")
    # parser.add_argument("--fine_grained_actions_map_file", type=str, default="/data/amerinov/data/holoassist/fine_grained_actions_map.txt")

    parser.add_argument("--dataset_name", type=str, default="holoassist")
    parser.add_argument("--base_model", type=str, default="InceptionV3")
    parser.add_argument("--fusion_mode", type=str, default="GSF")
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5, help="Number of transformations applied for single video clip to achieve better precision in evaluation.")
    parser.add_argument("--num_classes", type=int, default=2)
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
        dropout=args.dropout,
        verbose=False,
    ).to(device)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    div = model.div
    
    # Parallel!
    model = torch.nn.DataParallel(model).to(device)

    #  ========================= LOAD MODEL STATE =========================
    # 

    checkpoint = torch.load("checkpoints/holoassist_InceptionV3_GSF_mistake_00.pth", map_location=torch.device(device))
    model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=True)

    #  ========================= VALIDATION DATALOADER =========================
    # 

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
        transform=None, # We'll apply transformations later
        mode="validation"
    )

    # =====================================================================

    # TODO: calculate PRECISION and RECALL for 
    # mistake ground truths and for no mistake ground truths
    # by maximizing F1 score. Use threshold p < thr after SoftMax
    # (it is not clear how aauthors calculated F score, 
    # maybe this is an average of F1 scores or F-beta score?)

    # Store predictions and ground trues for metric calculation
    probs = torch.empty((len(va_dataset), 2), dtype=torch.float32)
    trues = torch.empty(len(va_dataset), dtype=torch.float32)

    # TODO: make work of parallel of batch and not one clip

    model.eval()
    with torch.no_grad():
        for clip_id in range(len(va_dataset)):
            
            # Get one clip.
            # Since we haven't applied transform yet,
            # va_x and va_y are not tensors
            va_x = va_dataset[clip_id][0] # List[PIL.Image.Image]
            va_y = va_dataset[clip_id][1] # int

            # Apply transformation args.repetitions times 
            # to achieve better precision in evaluation and 
            # make tensor out of stacked transformed video clips. 
            va_x_repeated = []
            for _ in range(args.repetitions):
                transformed_va_x = va_transform(va_x) # [t*c, h, w]
                va_x_repeated.append(transformed_va_x)
            va_x_repeated = torch.stack(va_x_repeated, dim=0) # [repeats, t*c, h, w]
            
            # Make tensor out of label and repeat it args.repetitions times.
            va_y = torch.tensor(va_y)
            va_y = torch.unsqueeze(va_y, dim=0)
            va_y_repeated = va_y.repeat(args.repetitions) # [repeats]

            # Make predictions for single clip (args.repetitions times)
            # and average predictions by repetitions
            va_x_repeated = va_x_repeated.to(device) 
            va_y_repeated = va_y_repeated.to(device) 

            clip_logits = model(va_x_repeated) # [repeats, num_classes]
            clip_logits = torch.mean(clip_logits, dim=0, keepdim=False) # [num_classes]
            # TODO: probabiblity calibration?
            clip_probs = F.softmax(clip_logits, dim=0)
            
            # Calculate metrics
            probs[clip_id] = clip_probs
            trues[clip_id] = va_y

            if clip_id > 0 and clip_id % 10 == 0:

                # Find threshold that maximize F1 score.
                thr, precision, recall, f1score = find_best_f1score(
                    trues=trues[:clip_id], 
                    probs=probs[:clip_id],
                )

                # Use this threshold to find precision, recall, f1score
                # for true mistake-only clips and for true correct-only clips.
                mistake_precision, mistake_recall, mistake_f1score = calc_metrics_by_threshold(
                    thr=thr,
                    trues=trues[:clip_id][trues[:clip_id] == 1],
                    probs=probs[:clip_id][trues[:clip_id] == 1],
                )
                correct_precision, correct_recall, correct_f1score = calc_metrics_by_threshold(
                    thr=thr,
                    trues=trues[:clip_id][trues[:clip_id] == 0],
                    probs=probs[:clip_id][trues[:clip_id] == 0],
                )

                print(f"clip_id={clip_id:04d}/{len(va_dataset):04d}",
                      f"clips_mistake={(trues[:clip_id] == 1).sum()}",
                      f"clips_correct={(trues[:clip_id] == 0).sum()}",
                      f"|",
                      f"thr={thr:.3f}",
                      f"precision={precision:.3f}",
                      f"recall={recall:.3f}",
                      f"f1score={f1score:.3f}",
                      f"|",
                      f"correct_precision={correct_precision:.3f}",
                      f"correct_recall={correct_recall:.3f}",
                      f"correct_f1score={correct_f1score:.3f}",
                      f"|",
                      f"mistake_precision={mistake_precision:.3f}",
                      f"mistake_recall={mistake_recall:.3f}",
                      f"mistake_f1score={mistake_f1score:.3f}",
                      flush=True)
                
                print(trues[:clip_id])
                print(probs[:clip_id])
                print((probs[:clip_id, 1] >= thr).int())
                print()

            if clip_id == 200:
                break

    print("DONE.")
