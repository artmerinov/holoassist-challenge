import torch
from typing import List
import numpy as np
from numpy.random import randint
import av


def temporal_sampling(
        frames: List[av.video.frame.VideoFrame],
        num_segments: int,
        mode: str = "train",
    ) -> List[av.video.frame.VideoFrame]:
    """
    Given the list of frames, sample `num_samples` frames.
    """
    num_frames = len(frames)
    average_duration = num_frames // num_segments

    if mode != "test":
        if average_duration > 0:
            segment_indices = (
                np.multiply(list(range(num_segments)), average_duration) + 
                randint(average_duration, size=num_segments)
            )
        else:
            segment_indices = np.zeros((num_segments,), dtype="int64")
    else:
        raise NotImplementedError()
    
    frames =  [frames[i] for i in segment_indices]
    return frames
