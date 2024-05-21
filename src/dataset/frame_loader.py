import os
import torch
from typing import List, Any
import av
import gc
from PIL import Image

from .decoder import pyav_decode_stream
from .video_container import get_video_container
from .utils import secs_to_pts, secs_to_pts


def load_av_frames_from_video(
        path_to_video: str, 
        start_secs: float, 
        end_secs: float
    ) -> List[av.video.frame.VideoFrame]: 
    """
    Load frames from a video between start and end time.
    """
    container = get_video_container(
        path_to_video=path_to_video
    )
    stream = container.streams.video[0]

    start_pts = secs_to_pts(
        time_in_seconds=start_secs,
        time_base=stream.time_base,
        start_pts=stream.start_time,
        round_mode="ceil",
    )
    end_pts = secs_to_pts(
        time_in_seconds=end_secs,
        time_base=stream.time_base,
        start_pts=stream.start_time,
        round_mode="ceil",
    )
    frames = pyav_decode_stream(
        container=container,
        start_pts=start_pts,
        end_pts=end_pts,
        stream=stream,
        buffer_size=0,
    )

    # To avoid memory leakage, it is impotant to close 
    # the container with "container.close()" and execute 
    # garbage collector "gc.collect()" later. 

    container.close()
    gc.collect()

    return frames


def transform_av_frames_to_PIL(
        frames: List[av.video.frame.VideoFrame],
    ) -> List[Image.Image]:
    """
    Transforms PyAv images to PIL images.
    """
    pil_images = [frame.to_image() for frame in frames]
    return pil_images


def load_frames_from_dir(
        holoassist_dir: str,
        video_name: str, 
        start_secs: float,
        end_secs: float,
        fps: int
    ):
    """
    Load images from directory and filter them 
    based on the time interval.
    """
    path_to_frames = os.path.join(holoassist_dir, video_name, "Export_py", "Video", "images")
    frames = sorted(os.listdir(path_to_frames))
    
    # Convert seconds to frames
    start_frame_index = int(start_secs * fps)
    end_frame_index = int(end_secs * fps)
    if start_frame_index == end_frame_index and end_frame_index < len(frames) - 1:
        end_frame_index += 1
    elif start_frame_index == end_frame_index and end_frame_index == len(frames) - 1:
        start_frame_index -= 1

    # Load only images within the specified frame range
    filt_frames = frames[start_frame_index:end_frame_index]
    filt_paths = [f"{path_to_frames}/{f}" for f in filt_frames]

    if len(filt_paths) == 0:
        print(f"len(frames)={len(frames)}",
            f"video_name={video_name}",
            f"start_frame_index={start_frame_index}",
            f"end_frame_index={end_frame_index}",
            flush=True)
        raise ValueError("HELLO")

    return filt_paths