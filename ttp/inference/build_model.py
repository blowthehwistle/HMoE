import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple, List

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from torchtitan.tools import utils
from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.tools.logging import init_logger, logger
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.utils import device_module, device_type
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.components.checkpoint import MODEL
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.components.metrics import DeviceMemoryMonitor, build_device_memory_monitor
from ttp.experiments.hmoe.infra.parallelize import apply_tp_minus_sp, apply_fsdp


from ttp.config.job_config import TTPJobConfig
from ttp.patches import register_models, run_patches
from ttp.experiments.hmoe.infra.parallelize import apply_tp_minus_sp

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# patches and register
run_patches()
register_models()

def find_last_step(folder: str) -> int:
    pattern = r"step-(\d+)"
    step_counts = []

    if not os.path.isdir(folder):
        return -1

    for filename in os.listdir(folder):
        match = re.search(pattern, filename)
        metadata_probe = os.path.join(folder, filename, ".metadata")
        if match and os.path.isfile(metadata_probe):
            step_counts.append(int(match.group(1)))
    if not step_counts:
        return -1
    return max(step_counts)

def init_config(config_path):

    init_logger()

    # Load configuration from toml file
    config_manager = ConfigManager(TTPJobConfig)
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    return config


def load_model_checkpoint(model: nn.Module, checkpoint_path):
    state_dict = model.state_dict()

    # Checkpoint Loading
    begin = time.monotonic()

    logger.info(f"Loading chkpt at: {checkpoint_path}")
    try:
        dcp.load(state_dict, checkpoint_id=checkpoint_path)
    except:  # noqa: E722 FIX ray log bug
        raise RuntimeError(traceback.print_exc())
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")


def build_tokenizer_and_model(
    device_memory_monitor: DeviceMemoryMonitor,
    config: TTPJobConfig,
    checkpoint_path: str | None = None,
    tp_size: int = 1,
    dp_shard: int = 1,
) -> Tuple[Tokenizer, nn.Module]:
    device_module, device_type = utils.device_module, utils.device_type
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f"{device_type}:{local_rank}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    train_spec = get_train_spec(config.model.name)

    logger.info(f"Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    tokenizer = train_spec.build_tokenizer_fn(config)

    model_cls = train_spec.model_cls
    if config.model.flavor.endswith("_varlen"):
        flavor = config.model.flavor.replace("_varlen", "")
    else:
        flavor = config.model.flavor

    model_args = train_spec.model_args[flavor]
    model_args.update_from_config(config)
    cpu_offload = config.inference.enable_cpu_offload
    reshard_after_forward_policy = config.inference.fsdp_reshard_after_forward

    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls(model_args)

    # Init distributed env
    if world_size > 1:
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
        )
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=dp_shard,
            cp=1,
            tp=tp_size,
            pp=1,
            ep=1,
            etp=1,
            world_size=tp_size * dp_shard,
        )
        # Build world mesh for parallelism
        world_mesh = parallel_dims.world_mesh
        if tp_size > 1:
            apply_tp_minus_sp(model, world_mesh["tp"], False, True)

        logger.info(f"world_mesh: {world_mesh}")

        if dp_shard > 1:
            if parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate",)
            else:
                dp_mesh_dim_names = ("dp_shard_cp",)
            dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
            logger.info(f"dp_mesh: {dp_mesh}")
            apply_fsdp(
                model,
                dp_mesh,
                param_dtype=TORCH_DTYPE_MAP[config.inference.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[config.inference.mixed_precision_reduce],
                pp_enabled=parallel_dims.pp_enabled,
                cpu_offload=cpu_offload,
                reshard_after_forward_policy=reshard_after_forward_policy,
            )

    else:
        world_mesh = None

    dist_utils.set_determinism(
        world_mesh,
        device,
        config.training.seed,
        config.training.deterministic
    )

    # materalize model
    if cpu_offload and dp_shard > 1:
        model.to_empty(device="cpu")
    else:
        model.to_empty(device=device)
    model.eval()

    with torch.device(device):
        model.freqs_cis = model._precompute_freqs_cis()

    if checkpoint_path is None:
        if config.inference.checkpoint_path is None:
            checkpoint_folder = os.path.join(config.job.dump_folder, config.checkpoint.folder)
            if config.checkpoint.load_step < 0:
                load_step = find_last_step(checkpoint_folder)
            else:
                load_step = config.checkpoint.load_step
            assert load_step >= 0
            checkpoint_path = os.path.join(checkpoint_folder, f"step-{load_step}")
        else:
            checkpoint_path = config.inference.checkpoint_path

    load_model_checkpoint(model, checkpoint_path)

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{utils.device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    return tokenizer, model
