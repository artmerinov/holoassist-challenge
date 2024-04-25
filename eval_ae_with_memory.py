import numpy as np
import argparse
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from src.utils.meters import AverageMeter
from src.utils.reproducibility import make_reproducible
from src.models.conv_ae_with_memory import ConvAEMemory
from src.dataset.video_dataset_ae import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Testing recognition error on validation set")
    parser.add_argument('--holoassist_dir', type=str, default="/data/amerinov/data/holoassist")
    parser.add_argument('--raw_annotation_file', type=str, default="/data/amerinov/data/holoassist/data-annotation-trainval-v1_1.json")
    parser.add_argument('--split_dir', type=str, default="/data/amerinov/data/holoassist/data-splits-v1")
    parser.add_argument('--fine_grained_actions_map_file', type=str, default="/data/amerinov/data/holoassist/fine_grained_actions_map.txt")
    parser.add_argument('--dataset_name', type=str, default="holoassist")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_segments', type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
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
        mode="validation",
        anomaly_mode="both",
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
        mode="validation"
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        drop_last=False, 
        pin_memory=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAEMemory(input_channels=args.num_segments * 3).to(device)
    model = torch.nn.DataParallel(model).to(device)

    checkpoint = torch.load('checkpoints/AE_02.pth', map_location=torch.device(device))
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])

    memory = np.load('checkpoints/memory_02.npy')
    memory = torch.Tensor(memory).to(device)
    
    model.eval();
    with torch.no_grad():

        rec_loss_normal_list = []
        rec_loss_anomaly_list = []
        anomaly_clips = 0
        normal_clips = 0

        for batch_id, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)

            x_hat, queries, augmented_queries, memory, score_query, score_memory, gathering_loss, spreading_loss = model(x, memory, train=False)

            # Select anomaly and normal
            anomaly_flg = y == 1
            normal_flg = y == 0

            # Anomaly reconstraction loss
            if anomaly_flg.sum() > 0:
                item_loss = torch.mean((x[anomaly_flg] - x_hat[anomaly_flg])**2)
                rec_loss_anomaly_list.append(item_loss.item())
                anomaly_clips += anomaly_flg.sum()
            
            # Normal reconstraction loss
            if normal_flg.sum() > 0:
                item_loss = torch.mean((x[normal_flg] - x_hat[normal_flg])**2)
                rec_loss_normal_list.append(item_loss.item())
                normal_clips += normal_flg.sum()
            
            if batch_id % 10 == 0:
                print(
                    f"batch_id={batch_id:04d}/{len(dataloader):04d}",
                    f"|",
                    f"anomaly_cnt={anomaly_clips}",
                    f"anomaly_median={np.median(rec_loss_anomaly_list):.2f}",
                    f"anomaly_mean={np.mean(rec_loss_anomaly_list):.2f}",
                    f"|",
                    f"normal_cnt={normal_clips}",
                    f"normal_median={np.median(rec_loss_normal_list):.2f}",
                    f"normal_mean={np.mean(rec_loss_normal_list):.2f}",
                    flush=True)