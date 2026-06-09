import os
from contextlib import nullcontext

import torch


def nvtx_range(message: str):
    """Return an opt-in NVTX range for Nsight Systems structure profiling."""
    if os.environ.get("TTP_NVTX_PROFILE") == "1" and torch.cuda.is_available():
        return torch.cuda.nvtx.range(message)
    return nullcontext()
