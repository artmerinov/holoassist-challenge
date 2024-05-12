import numpy as np
from numpy.random import randint
import av
from typing import Tuple, List


def temporal_sampling(
        frames: List[av.video.frame.VideoFrame],
        num_segments: int,
        mode: str = "train",
    ) -> Tuple[np.ndarray, List[av.video.frame.VideoFrame]]:
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

    # Since other modalities might have different frequencies,
    # we will also output the time portions of sampled frames.
    # So, we can easily select syncronised data from other modalities 
    # based on this sampling.
    sampling_portions = segment_indices / num_frames
    
    return sampling_portions, frames