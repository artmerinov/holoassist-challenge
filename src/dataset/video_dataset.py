import os
import torch
import numpy as np
import random
from numpy.random import randint
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Tuple, Any, List, Union, Iterable, Callable, Literal

from .prepare_annotation import prepare_annotation_action, prepare_annotation_mistake
from .prepare_split_list import get_video_name_list
from .frame_loader import load_av_frames_from_video, transform_av_frames_to_PIL, load_frames_from_dir
from .temporal_sampling import temporal_sampling
from .utils import fganame_to_fgaid
from .utils import make_video_path


def prepare_clips_data(
        raw_annotation_file: str,
        holoassist_dir: str,
        split_dir: str,
        fga_map_file: str,
        mode: Literal["train", "validation", "test"] = "train",
        task: Literal["action", "mistake"] = "action",
        debug: bool = False,
    ) -> Tuple[np.ndarray, ...]:

    all_video_names = get_video_name_list(
        split_dir=split_dir, 
        holoassist_dir=holoassist_dir, 
        mode=mode
    )
    if debug == True:
        all_video_names = all_video_names[::10]

    if task == "mistake":
        annotation = prepare_annotation_mistake(raw_annotation_file=raw_annotation_file)

        video_name_list, start_list, end_list, label_list = [], [], [], []
        for video_name in all_video_names:
            clips = annotation[video_name]
            for clip in clips:
                video_name_list.append(video_name)
                start_list.append(clip["start"])
                end_list.append(clip["end"])
                label_list.append(clip["label"])

    if task == "action":
        annotation = prepare_annotation_action(raw_annotation_file=raw_annotation_file)
        fganame_to_fgaid_dict = fganame_to_fgaid(fga_map_file=fga_map_file)

        video_name_list, start_list, end_list, label_list = [], [], [], []
        for video_name in all_video_names:
            clips = annotation[video_name]
            for clip in clips:
                video_name_list.append(video_name)
                start_list.append(clip["start"])
                end_list.append(clip["end"])
                label_list.append(fganame_to_fgaid_dict[clip["label"]])

    video_name_arr = np.array(video_name_list, dtype=np.string_)
    start_arr = np.array(start_list, dtype=np.float32)
    end_arr = np.array(end_list, dtype=np.float32)
    label_arr = np.array(label_list, dtype=np.int64)
    
    return video_name_arr, start_arr, end_arr, label_arr


class VideoDataset(Dataset):
    def __init__(
            self,
            holoassist_dir: str,
            video_name_arr: np.ndarray,
            start_arr: np.ndarray,
            end_arr: np.ndarray,
            label_arr: np.ndarray,
            num_segments: int = 8,
            transform: Callable = None,
            mode: Literal["train", "validation", "test"] = "train",
            use_hands: bool = False,
            fps: int = 10,
        ) -> None:

        self.holoassist_dir = holoassist_dir
        self.video_name_arr = video_name_arr
        self.start_arr = start_arr
        self.end_arr = end_arr
        self.label_arr = label_arr
        self.num_segments = num_segments
        self.transform = transform
        self.mode = mode
        self.use_hands = use_hands
        self.fps = fps

    def __getitem__(self, index):

        path_to_video = make_video_path(
            holoassist_dir=self.holoassist_dir, 
            video_name=self.video_name_arr[index].decode()
        )
        start = self.start_arr[index]
        end = self.end_arr[index]
        label = self.label_arr[index]

        # Extract frames from video using start and end time. 
        frames = load_av_frames_from_video(
            path_to_video=path_to_video, 
            start_secs=start, 
            end_secs=end
        )

        # Perform temporal sampling
        sampling_portions, frames = temporal_sampling(
            frames=frames,
            num_segments=self.num_segments,
            # mode=self.mode
        )
        
        # Transfrom list of PyAv images to list of PIL images
        frames = transform_av_frames_to_PIL(frames=frames)
        
        # Perform spatial sampling and apply transformations
        if self.transform:
            frames = self.transform(frames)

        return frames, label

    # def __getitem__(self, index):

    #     video_name = self.video_name_arr[index].decode()
    #     start = self.start_arr[index]
    #     end = self.end_arr[index]
    #     label = self.label_arr[index]

    #     # Load frames from directory
    #     frame_paths = load_frames_from_dir(
    #         holoassist_dir=self.holoassist_dir,
    #         video_name=video_name,
    #         start_secs=start,
    #         end_secs=end,
    #         fps=self.fps,
    #     )

    #     # Perform temporal sampling
    #     sampling_portions, frame_paths = temporal_sampling(
    #         frames=frame_paths,
    #         num_segments=self.num_segments,
    #         mode=self.mode
    #         )

    #     # Create list of PIL images based on frame paths
    #     frames = []
    #     for path in frame_paths:
    #         frame = Image.open(path)
    #         frame = frame.convert('RGB')
    #         frames.append(frame)
        
    #     # Perform spatial sampling and apply transformations
    #     if self.transform:
    #         frames = self.transform(frames)

    #     return frames, label

    def __len__(self):
        return len(self.video_name_arr)
    

# def prepare_clips_data_test(
#         test_action_clips_file: str
#     ) -> Tuple[np.ndarray, ...]:
    
#     key_list = []
#     video_name_list = []
#     start_list = []
#     end_list = []

#     with open(test_action_clips_file, 'r') as f:
#         for line in f:

#             record = line.strip().split('_')
#             video_name = " ".join(record[:-2])
#             start = float(record[-2])
#             end = float(record[-1])

#             key_list.append(line.strip())
#             video_name_list.append(video_name)
#             start_list.append(start)
#             end_list.append(end)

#     key_arr = np.array(key_list, dtype=np.string_)
#     video_name_arr = np.array(video_name_list, dtype=np.string_)
#     start_arr = np.array(start_list, dtype=np.float32)
#     end_arr = np.array(end_list, dtype=np.float32)

#     return key_arr, video_name_arr, start_arr, end_arr


def prepare_clips_data_test(
        holoassist_dir: str,
        test_action_clips_file: str
    ):
    
    key_list = []
    video_name_list = []
    start_list = []
    end_list = []

    successful_clips = 0
    failed_clips = 0
    successful_video_names = set()
    failed_video_names = set()

    with open(test_action_clips_file, 'r') as f:
        for line in f:

            key = line.strip()
            record = key.split('_')
            video_name = "_".join(record[:-2])
            start = float(record[-2])
            end = float(record[-1])

            # Checck if the path to video exists:
            path_to_video = make_video_path(
                holoassist_dir=holoassist_dir, 
                video_name=video_name
            )
            if os.path.exists(path_to_video):
                
                key_list.append(key)
                video_name_list.append(video_name)
                start_list.append(start)
                end_list.append(end)

                successful_video_names.add(video_name)
                successful_clips += 1

            else:
                failed_video_names.add(video_name)
                failed_clips += 1
    
    print(f"Number of successful fine-grained clips: {successful_clips}",
          f"\nNumber of failed fine-grained clips: {failed_clips}", 
          f"\nNumber of successful videos: {len(successful_video_names)}",
          f"\nNumber of failed videos: {len(failed_video_names)}",
          f"\nFailed video names are: {failed_video_names}", 
          flush=True)

    video_name_arr = np.array(video_name_list, dtype=np.string_)
    start_arr = np.array(start_list, dtype=np.string_)
    end_arr = np.array(end_list, dtype=np.string_)

    return key_list, video_name_arr, start_arr, end_arr


class VideoDatasetTest(Dataset):
    def __init__(
            self,
            holoassist_dir: str,
            video_name_arr: np.ndarray,
            start_arr: np.ndarray,
            end_arr: np.ndarray,
            num_segments: int,
            transform: Callable,
        ) -> None:

        self.holoassist_dir = holoassist_dir
        self.video_name_arr = video_name_arr
        self.start_arr = start_arr
        self.end_arr = end_arr
        self.num_segments = num_segments
        self.transform = transform

    def __getitem__(self, index):

        video_name = self.video_name_arr[index].decode()
        start = self.start_arr[index].decode()
        end = self.end_arr[index].decode()
        # key = f"{video_name}_{start}_{end}"

        start = float(start)
        end = float(end)

        path_to_video = make_video_path(
            holoassist_dir=self.holoassist_dir, 
            video_name=video_name
        )

        # Extract frames from video using start and end time. 
        frames = load_av_frames_from_video(
            path_to_video=path_to_video, 
            start_secs=start, 
            end_secs=end
        )

        # Perform temporal sampling
        sampling_portions, frames = temporal_sampling(
            frames=frames,
            num_segments=self.num_segments,
            # mode=self.mode
        )
        
        # Transfrom list of PyAv images to list of PIL images
        frames = transform_av_frames_to_PIL(frames=frames)

        # Perform spatial sampling and apply transformations
        if self.transform:
            frames = self.transform(frames)

        return frames
    
    def __len__(self):
        return len(self.video_name_arr)
    