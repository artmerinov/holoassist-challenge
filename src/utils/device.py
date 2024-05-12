import torch
from typing import List


def prepare_device(n_gpu_use: int, gpu_ids: List[int] = None):
    n_gpu_total = torch.cuda.device_count()
    
    if n_gpu_use > 0 and n_gpu_total == 0:
        n_gpu_use = 0
    if n_gpu_use > n_gpu_total:
        n_gpu_use = n_gpu_total

    if n_gpu_use > 0:
        device = torch.device(f"cuda:" + ','.join(str(x) for x in gpu_ids))
    else:
        device = torch.device("cpu")

    print("Executing on device: ", device)
    
    return device

