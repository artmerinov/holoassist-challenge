import numpy as np
import torch


def hands_temporal_sampling(
        data: torch.Tensor,
        sampling_portions: np.ndarray,
) -> torch.Tensor:
    """
    Make temporal sampling based on given sampling portions.

    Args:
        data: array of shape [T, ...]
        sampling_portions: array of time fractions.
    """
    num_records = data.shape[0]
    ids = [int(idx) for idx in sampling_portions*num_records]

    # Perform temporal sampling
    data = data[ids]

    return data
