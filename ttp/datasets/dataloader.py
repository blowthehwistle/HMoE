from torchtitan.components.tokenizer import Tokenizer
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.tools.logging import logger
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from ttp.datasets.mmap_datasets import build_mmap_dataloader
from ttp.config.job_config import TTPJobConfig


def build_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: TTPJobConfig
) -> ParallelAwareDataloader:
    dataset_type = job_config.training.dataset_type
    logger.info(f"Building dataset with type-{dataset_type} .")
    match dataset_type:
        case "hf":
            assert not job_config.sft.enable_sft
            return build_hf_dataloader(
                dp_world_size,
                dp_rank,
                tokenizer,
                job_config,
                infinite=True,
            )
        case "mmap":
            return build_mmap_dataloader(
                dp_world_size,
                dp_rank,
                tokenizer,
                job_config
            )
        case _:
            raise ValueError(f"Unsupported dataset type: {dataset_type}.")
