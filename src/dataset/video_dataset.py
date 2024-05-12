import torch
import numpy as np
import random
from numpy.random import randint
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Tuple, Any, List, Union, Iterable, Callable, Literal

from .prepare_annotation import prepare_annotation
from .prepare_split_list import get_video_name_list
from .video_record import ClipRecord
from .frame_loader import load_av_frames_from_video, transform_av_frames_to_PIL
from .temporal_sampling import temporal_sampling


def prepare_clips_data(
        raw_annotation_file: str,
        holoassist_dir: str,
        split_dir: str,
        fine_grained_actions_map_file: str,
        mode: Literal["train", "validation", "test"] = "train",
        debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    annotation = prepare_annotation(
        raw_annotation_file=raw_annotation_file
    )
    video_name_list = get_video_name_list(
        split_dir=split_dir, 
        holoassist_dir=holoassist_dir, 
        mode=mode
    )
    
    if debug:
        video_name_list = video_name_list[::100]
    
    # Find total number of clips in videos
    # considering given mode (train/validation/test)

    clips_num = 0
    for video_name in video_name_list:
        clips = annotation[video_name]
        for clip in clips:
            clips_num += 1
    print(f"Number of clips: {clips_num} for mode {mode}")
    
    # Create data structure for each clip
    # https://github.com/pytorch/pytorch/issues/13246

    clip_path_to_video_arr = []
    clip_start_arr = np.empty(clips_num, dtype=np.float32)
    clip_end_arr = np.empty(clips_num, dtype=np.float32)
    clip_action_id_arr = np.empty(clips_num, dtype=np.int64)
    clip_mistake_arr = np.empty(clips_num, dtype=np.int64)

    index = 0
    for video_name in video_name_list:
        clips = annotation[video_name]
        for clip in clips:
            clip_record = ClipRecord(
                holoassist_dir=holoassist_dir,
                video_name=video_name,
                fine_grained_actions_map_file=fine_grained_actions_map_file,
                clip=clip,
            )
            clip_path_to_video_arr.append(clip_record.path_to_video)
            clip_start_arr[index] = clip_record.start
            clip_end_arr[index] = clip_record.end
            clip_action_id_arr[index] = clip_record.action_id
            clip_mistake_arr[index] = clip_record.mistake

            index += 1

    clip_path_to_video_arr = np.array(clip_path_to_video_arr, dtype=np.string_)

    return clip_path_to_video_arr, clip_start_arr, clip_end_arr, clip_action_id_arr, clip_mistake_arr


class VideoDataset(Dataset):

    def __init__(
            self,  
            clip_path_to_video_arr: np.ndarray,
            clip_start_arr: np.ndarray,
            clip_end_arr: np.ndarray,
            clip_label_arr: np.ndarray,
            num_segments: int = 8,
            transform: Callable = None,
            mode: Literal["train", "validation", "test"] = "train",
        ) -> None:

        self.clip_path_to_video_arr = clip_path_to_video_arr
        self.clip_start_arr = clip_start_arr
        self.clip_end_arr = clip_end_arr
        self.clip_label_arr = clip_label_arr
        
        self.num_segments = num_segments
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):

        clip_path_to_video = self.clip_path_to_video_arr[index].decode()
        clip_start = self.clip_start_arr[index]
        clip_end = self.clip_end_arr[index]
        clip_label = self.clip_label_arr[index]

        # Extract frames from video using start and end time. 
        frames = load_av_frames_from_video(
            path_to_video=clip_path_to_video, 
            start_secs=clip_start, 
            end_secs=clip_end
        )

        # Perform temporal sampling
        frames = temporal_sampling(
            frames=frames,
            num_segments=self.num_segments,
            mode=self.mode
        )
        
        # Transfrom list of PyAv images to list of PIL images
        frames = transform_av_frames_to_PIL(frames=frames)
        
        # Perform spatial sampling and apply transformations
        if self.transform:
            frames = self.transform(frames)
        
        return frames, clip_label

    def __len__(self):
        return len(self.clip_path_to_video_arr)