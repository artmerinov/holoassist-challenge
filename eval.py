# import gc
# import os
# import time
# import numpy as np
# import random
# import argparse
# from datetime import datetime
# from functools import partial
# import torch
# import torch.nn as nn
# from torch.nn.utils import clip_grad_norm_
# from torchvision.transforms import Compose
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from einops.layers.torch import Rearrange

# from src.opts.opts import parser
# from src.utils.reproducibility import make_reproducible
# from src.models.model import VideoModel
# from src.dataset.video_dataset import VideoDataset, prepare_clips_data
# from src.dataset.video_transforms import (
#     GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupScale, 
#     GroupCenterCrop, GroupOverSample
# )
# from src.utils.meters import AverageMeter
# from src.utils.metrics import calc_accuracy


# if __name__ == "__main__":

#     # Reproducibility.
#     # Set up initial random states.
#     make_reproducible(random_seed=0)

#     parser = argparse.ArgumentParser(description="Testing on the full validation set")
#     parser.add_argument('--dataset_name', type=str, choices=["holoassist"])
#     parser.add_argument('--base_model', type=str, choices=["InceptionV3"])
#     parser.add_argument("--fusion_mode", type=str, default=None, choices=[None, "GSF"], help="Fusion mode")
#     parser.add_argument("--dropout", type=float, default=0.5, help="Dropout ratio of last fc layer of base model")
#     parser.add_argument('--checkpoint', 
#                         type=str, 
#                         default="/data/amerinov/projects/holoassist/checkpoints/holoassist_InceptionV3_GSF_06.pth", 
#                         help="Path to saved model checkpoint")
#     parser.add_argument('--num_segments', type=int)
#     parser.add_argument('--test_crops', type=int, default=1, help="Number crops of group of images for better precision")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in dataloader")
#     args = parser.parse_args()
#     print(args)

#     if args.dataset_name == 'holoassist':
#         num_classes = 1887 # actions
#     else:
#         raise NotImplementedError()
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = VideoModel(
#         num_classes=num_classes, 
#         num_segments=args.num_segments, 
#         base_model=args.base_model,
#         fusion_mode=args.fusion_mode,
#         dropout=args.dropout,
#     ).to(device)
#     # print(model)

#     crop_size = model.crop_size
#     scale_size = model.scale_size
#     input_mean = model.input_mean
#     input_std = model.input_std
#     learnable_named_parameters = model.learnable_named_parameters
#     model = torch.nn.DataParallel(model).to(device)

#     # Load state
#     checkpoint = torch.load(f=args.checkpoint, map_location=torch.device(device))
#     model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    
#     # ============================ VALIDATION DATALOADER ============================

#     va_clip_path_to_video_arr, va_clip_start_arr, va_clip_end_arr, va_clip_action_id_arr = prepare_clips_data(
#         raw_annotation_file=args.raw_annotation_file,
#         holoassist_dir=args.holoassist_dir, 
#         split_dir=args.split_dir,
#         fine_grained_actions_map_file=args.fine_grained_actions_map_file,
#         mode="validation",
#     )
#     va_transform = Compose([
#         GroupMultiScaleCrop(input_size=crop_size, scales=[1, .875]),
#         Stack(),
#         ToTorchFormatTensor(div=(args.base_model not in ['BNInception'])),
#         GroupNormalize(mean=input_mean, std=input_std),
#     ])
#     va_dataset = VideoDataset(
#         clip_path_to_video_arr=va_clip_path_to_video_arr,
#         clip_start_arr=va_clip_start_arr,
#         clip_end_arr=va_clip_end_arr,
#         clip_action_id_arr=va_clip_action_id_arr,
#         num_segments=args.num_segments,
#         transform=va_transform,
#         mode="validation"
#     )
#     va_dataloader = DataLoader(
#         dataset=va_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=False,
#         num_workers=args.num_workers, 
#         drop_last=False, 
#         pin_memory=False
#     )

#     # ============================ EVALULATION CYCLE ============================

#     va_epoch_acc1 = AverageMeter()
#     va_epoch_acc5 = AverageMeter()

#     model.eval()
#     with torch.no_grad():
#         for va_batch_id, va_batch in enumerate(va_dataloader):

#             # Since for each video we have some crop repeats,
#             # we have different batch shape:
#             # [n, r * t * c, h, w], where r is number of crops.
#             # Therefore, we need to reshape our input tensor 
#             # and repeat label r times.

#             va_x = va_batch[0]
#             va_y = va_batch[1].repeat(args.test_crops)
#             # print(va_x.size(), va_y.size())

#             n, r_t_c, h, w = va_x.size()
#             r = args.test_crops
#             t = args.num_segments
#             c = r_t_c // (r*t)
#             # print(n, r, t, c, h, w)
            
#             va_x = Rearrange("n (r t c) h w -> (n r) (t c) h w", 
#                              n=n, r=r, t=t, c=c, h=h, w=w)(va_x)
#             # print(va_x.size(), va_y.size())

#             # After this transformation we can work the same way as 
#             # we multiply number of batches r times
#             # va_x: [n=1 * r=10, t=16 * c=3, h=224, w=224]
#             # va_y: [n=1 * r=10]

#             va_x = va_x.to(device) 
#             va_y = va_y.to(device) 

#             # Make prediction for validation batch
#             va_preds = model(va_x).to(device) 
#             va_acc1, va_acc5 = calc_accuracy(va_preds, va_y, topk=(1,5))
#             va_epoch_acc1.update(value=va_acc1, n=args.batch_size)
#             va_epoch_acc5.update(value=va_acc5, n=args.batch_size)

#             if va_batch_id % 20 == 0:
#                 print(f"va_batch_id={va_batch_id:04d}/{len(va_dataloader):04d}",
#                       f"va_epoch_acc@1={va_epoch_acc1.avg:.3f}",
#                       f"va_epoch_acc@5={va_epoch_acc5.avg:.3f}",
#                       flush=True)
                
#     # TODO: confusion_matrix
                
#     print("DONE")
#     print(f"va_epoch_acc@1={va_epoch_acc1.avg:.3f}")
#     print(f"va_epoch_acc@5={va_epoch_acc5.avg:.3f}")