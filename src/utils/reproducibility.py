import os
import random
import numpy as np
import torch


def make_reproducible(random_seed: int) -> None:
    """
    Make experiments reproducible.
    """
    print(f"Making reproducible on seed {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # https://pytorch.org/docs/stable/notes/randomness.html
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/5
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(random_seed)
