import os
import argparse
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from einops.layers.torch import Rearrange

from src.utils.reproducibility import make_reproducible
from src.models.model import VideoModel
from src.dataset.video_dataset import VideoDataset, prepare_clips_data
from src.dataset.video_transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize


class Hook:
    def __init__(self, module, backward=False):

        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    
    def close(self):
        self.hook.remove()


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
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--checkpoint", type=str, help="Best model weigths.")
    args = parser.parse_args()
    print(args)

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
        mode="train",
        # mode="validation",
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
        # mode="validation",
        mode="train",
        task="action",
        debug=False,
    )

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
        # mode="validation",
        mode="train",
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

    processed_clips = 0
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
        
            xs = batch[0].to(device) # video batch with image sequences [n, t_c, h, w]
            ys = batch[1].to(device) # video batch labels [n]

            n, t_c, h, w = xs.size()
            t = args.num_segments 
            c = t_c // t

            # Make a hook to retrieve output of intermeedieate layer
            layer = model.base_model.top_cls_pool
            hook = Hook(module=layer)
            
            # Perform forward pass
            output = model(xs)

            # Retrieve features
            pool_output = hook.output.detach() # n t k=num_classes
            k = pool_output.size(1)
            pool_output = Rearrange("(n t) k 1 1 -> n t k", n=n, t=t, k=k)(pool_output)
            avg_feature = torch.mean(pool_output, dim=1)
            features.append(avg_feature)
            labels.append(ys)
            hook.close()

            processed_clips += xs.size(0)
            if batch_id % 100 == 0:
                print(f"batch_id={batch_id:04d}/{len(dataloader):04d}",
                        f"processed_clips={processed_clips:05d}/{len(dataset):05d}",
                        f"time={datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}",
                        flush=True)

    features_tensor = torch.cat(features, axis=0)
    labels_tensor = torch.cat(labels, axis=0)

    # Save results
    tsne_folder = 'tsne_artefacts'
    os.makedirs(tsne_folder, exist_ok=True)
    checkpoint_fn = args.checkpoint.split("/")[-1].split(".")[0]
    torch.save(features_tensor, f"{tsne_folder}/{checkpoint_fn}_features.pt")
    torch.save(labels_tensor, f"{tsne_folder}/{checkpoint_fn}_labels.pt")

    print("DONE.")
