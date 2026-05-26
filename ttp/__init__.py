import logging
import os
import torch
from ttp.patches import run_patches

_COMPATIBLE_TORCH_VERSIONS = [
    "2.9.0+main.de744ca4b19.post20250818",
]

if torch.__version__ not in _COMPATIBLE_TORCH_VERSIONS:
    message = (
        f"Compatible torch versions: {_COMPATIBLE_TORCH_VERSIONS}, "
        f"current torch version is {torch.__version__}."
    )
    if os.environ.get("TTP_STRICT_TORCH_VERSION") == "1":
        raise ValueError(message)
    logging.getLogger(__name__).warning(message)

run_patches()  # do patch first
