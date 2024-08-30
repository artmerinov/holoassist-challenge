import numpy as np
from numpy.random import randint
import av
from typing import Tuple, List, Any


def temporal_sampling(
        frames: List[Any],
        num_segments: int,
        # mode: str = "train",
    ) -> Tuple[np.ndarray, List[Any]]:
    """
    Given the list of frames, sample `num_samples` frames.
    """
    num_frames = len(frames)
    average_duration = num_frames // num_segments

    # if mode != "test":
    #     if average_duration > 0:
    #         segment_indices = (
    #             np.multiply(list(range(num_segments)), average_duration) + 
    #             randint(average_duration, size=num_segments)
    #         )
    #     elif num_frames > num_segments:
    #         segment_indices = np.sort(randint(num_frames, size=num_segments))
    #     else:
    #         segment_indices = np.zeros((num_segments,), dtype="int64")
    # else:
    #     raise NotImplementedError()

    # if average_duration > 0:
    #     segment_indices = (
    #         np.multiply(list(range(num_segments)), average_duration) + 
    #         randint(average_duration, size=num_segments)
    #     )
    # elif num_frames > num_segments:
    #     segment_indices = np.sort(randint(num_frames, size=num_segments))
    # else:
    #     segment_indices = np.zeros((num_segments,), dtype="int64")

    if average_duration > 1:
        segment_indices = (
            np.multiply(list(range(num_segments)), average_duration) + 
            randint(average_duration, size=num_segments)
        )
    else:
        segment_indices = np.sort(randint(num_frames, size=num_segments))
    
    frames =  [frames[i] for i in segment_indices]

    # Since other modalities might have different frequencies,
    # we will also output the time portions of sampled frames.
    # So, we can easily select syncronised data from other modalities 
    # based on this sampling.
    sampling_portions = segment_indices / num_frames
    
    return sampling_portions, frames