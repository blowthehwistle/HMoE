import subprocess
import torchtitan.tools.utils
from torchtitan.tools.logging import logger
from ttp.patches import as_patch


torchtitan_get_peak_flops = torchtitan.tools.utils.get_peak_flops


@as_patch
def get_peak_flops(device_name: str) -> int:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "H20" in device_name:
        return 147e12
    else:
        return torchtitan_get_peak_flops(device_name)


def do_patch():
    torchtitan.tools.utils.get_peak_flops = get_peak_flops
