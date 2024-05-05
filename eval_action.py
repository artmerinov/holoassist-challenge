import argparse
import torch
from torchvision.transforms import Compose

from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from src.utils.meters import AverageMeter
from src.utils.metrics import calc_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # local run
    # parser.add_argument("--holoassist_dir", type=str, default="/Users/artemmerinov/data/holoassist")
    # parser.add_argument("--raw_annotation_file", type=str, default="/Users/artemmerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    # parser.add_argument("--split_dir", type=str, default="/Users/artemmerinov/data/holoassist/data-splits-v1")
    # parser.add_argument("--fine_grained_actions_map_file", type=str, default="/Users/artemmerinov/data/holoassist/fine_grained_actions_map.txt")

    # server run
    parser.add_argument("--holoassist_dir", type=str, default="/data/amerinov/data/holoassist")
    parser.add_argument("--raw_annotation_file", type=str, default="/data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    parser.add_argument("--split_dir", type=str, default="/data/amerinov/data/holoassist/data-splits-v1")
    parser.add_argument("--fine_grained_actions_map_file", type=str, default="/data/amerinov/data/holoassist/fine_grained_actions_map.txt")

    parser.add_argument("--dataset_name", type=str, default="holoassist")
    parser.add_argument("--base_model", type=str, default="InceptionV3")
    parser.add_argument("--fusion_mode", type=str, default="GSF")
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5, help="Number of transformations applied for single video clip to achieve better precision in evaluation.")
    parser.add_argument("--num_classes", type=int, default=1887)
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

    checkpoint = torch.load("checkpoints/holoassist_InceptionV3_GSF_10.pth", map_location=torch.device(device))
    model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=True)

    #  ========================= VALIDATION DATALOADER =========================
    # 

    va_clip_path_to_video_arr, va_clip_start_arr, va_clip_end_arr, va_clip_action_id_arr, _ = prepare_clips_data(
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
        clip_label_arr=va_clip_action_id_arr,
        num_segments=args.num_segments,
        transform=None, # We'll apply transformations later
        mode="validation"
    )

    # =====================================================================

    va_acc1 = AverageMeter()
    va_acc5 = AverageMeter()

    # TODO: maake work of parallel of abtch and not one clip

    model.eval()
    with torch.no_grad():
        for va_clip_id in range(len(va_dataset)):
            
            # Get one clip.
            # Since we haven't applied transform yet,
            # va_x and va_y are not tensors
            va_x = va_dataset[va_clip_id][0] # List[PIL.Image.Image]
            va_y = va_dataset[va_clip_id][1] # int

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
            va_y = va_y.to(device)
            va_y_repeated = va_y.repeat(args.repetitions) # [repeats]

            # Make predictions for single clip (args.repetitions times)
            # and average predictions by repetitions
            va_x_repeated = va_x_repeated.to(device) 
            va_y_repeated = va_y_repeated.to(device) 

            va_preds = model(va_x_repeated) # [repeats, num_classes]
            va_preds = torch.mean(va_preds, dim=0, keepdim=True) # [1, num_classes]
                
            va_clip_acc1, va_clip_acc5 = calc_accuracy(va_preds, va_y, topk=(1, 5))
            va_acc1.update(va_clip_acc1, n=1)
            va_acc5.update(va_clip_acc5, n=1)

            if va_clip_id % 10 == 0:
                print(f"va_clip_id={va_clip_id:04d}/{len(va_dataset):04d}",
                      f"va_acc@1={va_acc1.avg:.3f}",
                      f"va_acc@5={va_acc5.avg:.3f}",
                      flush=True)
                
    print("DONE.")