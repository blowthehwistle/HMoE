from torchtitan.components.tokenizer import Tokenizer
from ttp.components.dataloader import ParallelAwareDataloader
from ttp.config.job_config import TTPJobConfig
from ttp.datasets.mmap.dataset import load_mmap_dataset


def build_mmap_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,  # noqa
    job_config: TTPJobConfig
):
    if job_config.training.mmap_only_one_epoch:
        num_samples = None
    else:
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            global_batch_size = job_config.training.local_batch_size * dp_world_size
        num_samples = int(job_config.training.steps * global_batch_size)

    mmap_ds = load_mmap_dataset(
        data_name=job_config.training.dataset.lower(),
        data_prefix=job_config.training.dataset_path,
        num_samples=num_samples,
        seq_length=job_config.training.seq_len,
        shuffle_doc_idx=job_config.training.mmap_shuffle_doc_idx,
        shuffle_sample_idx=job_config.training.mmap_shuffle_sample_idx,
        seed=job_config.training.seed,
        sft=job_config.sft.enable_sft,
        sft_loss_mask_token_minimum_id=job_config.sft.loss_mask_token_minimum_id,
        loss_mask_token_ids=job_config.training.loss_mask_token_ids,
        sft_loss_mask_token_ids=job_config.sft.sft_loss_mask_token_ids,
    )

    return ParallelAwareDataloader(
        dataset=mmap_ds,
        batch_sampler_offset=job_config.training.mmap_batch_sampler_offset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        num_workers=job_config.training.mmap_dataloader_num_workers
    )
