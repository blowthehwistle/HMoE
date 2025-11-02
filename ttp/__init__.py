import torch
from ttp.patches import run_patches

_COMPATIBLE_TORCH_VERSIONS = [
    "2.9.0+main.de744ca4b19.post20250818",
]

if torch.__version__ not in _COMPATIBLE_TORCH_VERSIONS:
    raise ValueError(
        f"Compatible torch versions: {_COMPATIBLE_TORCH_VERSIONS}, "
        f"current torch verison is {torch.__version__}."
    )

run_patches()  # do patch first
