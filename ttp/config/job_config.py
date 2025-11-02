from dataclasses import dataclass, field
from torchtitan.config import JobConfig, Model, Training
from ttp.protocols.train_spec import NOT_SET
from typing import List, Any, Literal


@dataclass
class TTPModel(Model):
    tokenizer_type: str = "hf_tokenizer"  # hf_tokenizer / hunyuan_tokenizer
    ssm_cp_version: int = 1  # only support by Hybrid
    vocab_size: int = 122880
    eos_id: int = 120020


@dataclass
class TTPTraining(Training):
    type_name: str = "ttp"  # ttp / titan
    dataset_type: str = "hf"  # hf / mmap
    loss_function_type: str = "fused_cross_entropy"  # cross_entropy / fused_cross_entropy
    mtp_loss_weights: List[Any] = NOT_SET
    rope_theta: float | None = None
    mmap_batch_sampler_offset: int = 0
    mmap_shuffle_doc_idx: bool = True
    mmap_shuffle_sample_idx: bool = True
    mmap_dataloader_num_workers: int = 0
    mmap_only_one_epoch: bool = False
    loss_mask_token_ids: List[int] | None = None  # tokens to be masked for loss calculation during pretraining


@dataclass
class TTPInference:
    mixed_precision_param: Literal["bfloat16", "float32"] = "float32"
    mixed_precision_reduce: Literal["bfloat16", "float32"] = "float32"
    checkpoint_path: str | None = None
    use_inference_caches: bool = True
    enable_cpu_offload: bool = False  # following fsdp, move model weights to cpu
    enable_kvcache_cpu_offload: bool = False  # move kv cache cpu, slower, but less gpu memory
    fsdp_reshard_after_forward: str = "default"  # default / always / never


@dataclass
class TTPSFT:
    enable_sft: bool = False  # whether to use sft loss
    loss_mask_token_minimum_id: int = 300000  # sft loss mask token id
    sft_loss_mask_token_ids: List[int] | None = None  # tokens to be masked for loss calculation during sft
    varlen_token_id: int = 120026  # sft eod id


@dataclass
class TTPJobConfig(JobConfig):
    model: TTPModel = field(default_factory=TTPModel)
    training: TTPTraining = field(default_factory=TTPTraining)
    sft: TTPSFT = field(default_factory=TTPSFT)
    inference: TTPInference = field(default_factory=TTPInference)
