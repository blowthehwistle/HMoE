from torch import nn
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger
from ttp.config.job_config import TTPJobConfig
from ttp.protocols.train_spec import NOT_SET
from dataclasses import dataclass
from typing import List


@dataclass
class HybridModelArgs(BaseModelArgs):
    dim: int = 4096
    n_heads: int = 16
    n_kv_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int = NOT_SET  # should be setting by job_config
    eos_id: int = NOT_SET  # should be setting by job_config
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None  # MoE or (FFN if ffn_hidden_dim is None), 8/3 * ffn_dim_multiplier * dim
    ffn_hidden_dim: int | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 32768
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    qk_norm: bool = True  # use qk normalization in attention block
    qk_norm_after_rope: bool = True  # use qk normalization after rotary embedding
    initializer_range: float = 0.02

    # moe config
    num_experts: int = 8  # will not work for heterogeneous moe
    num_shared_experts: int = 1
    top_k: int = 1
    use_grouped_mm: bool = True
    load_balance_coeff: float | None = None
    enable_log_expert_bias: bool = False
    moe_router_use_sigmoid: bool = False
    expert_hidden_dim: int = 768  # will not work for heterogeneous moe
    route_norm: bool = True
    aux_load_balance_loss_coeff: int | None = None
    p_penalty_coeff: float | None = None

    # heterogeneous moe config
    use_heterogeneous_moe: bool = False
    expert_hidden_dims: List[int] | None = None
    num_expert_groups: int = 1
    num_experts_per_group: int = 8
    shared_expert_hidden_dim: int | None = None

    # hybrid config
    hybrid_config: str = "Af-Af"  # "-" == ""

    @property
    def n_layers(self) -> int:
        return len(self.hybrid_patterns)

    @property
    def hybrid_patterns(self) -> List[str]:
        return list(map(lambda x: x.lower(), self.hybrid_config.replace("-", "")))

    def update_from_config(self, job_config: TTPJobConfig) -> None:
        self.max_seq_len = max(self.max_seq_len, job_config.training.seq_len)

        self.vocab_size = job_config.model.vocab_size
        logger.info(f"Vocab size: {self.vocab_size}")
        assert self.vocab_size % 8192 == 0, f"vocab_size {self.vocab_size} is not multiple of 8192"

        if job_config.sft.enable_sft:
            self.eos_id = job_config.sft.varlen_token_id
        else:
            self.eos_id = job_config.model.eos_id
        logger.info(f"EOS ID: {self.eos_id}")

        if job_config.training.rope_theta is not None:
            self.rope_theta = float(job_config.training.rope_theta)

        self.ssm_cp_version = job_config.model.ssm_cp_version
        self.mtp_loss_weights = job_config.training.mtp_loss_weights

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.head_dim is None:
            assert self.dim % self.n_heads == 0, f"dim={self.dim}, n_heads={self.n_heads}"
            self.head_dim = self.dim // self.n_heads

        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = int(8 * self.dim / 3)
            # custom dim factor multiplier
            if self.ffn_dim_multiplier is not None:
                self.ffn_hidden_dim = int(self.ffn_dim_multiplier * self.ffn_hidden_dim)
            self.ffn_hidden_dim = self.multiple_of * ((self.ffn_hidden_dim + self.multiple_of - 1) // self.multiple_of)
        else:
            assert self.ffn_hidden_dim % self.multiple_of == 0, \
                f"ffn_hidden_dim={self.ffn_hidden_dim}, multiple_of={self.multiple_of}"

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())

        # TODO: add flops calculation

        return nparams, None
