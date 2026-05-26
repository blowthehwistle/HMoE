import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function
from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger
from ttp.experiments.cache import InferenceCache, AttentionKVCache
from .args import HybridModelArgs
from ttp.components.metrics import push_extra_metrics, MetricsAvgAccumulatorHook
from typing import List
from .moe import MoE
from .hmoe import HeterogeneousMoE


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1, f"ndim={ndim}"
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1]), f"freqs_cis={freqs_cis.shape} vs ({seqlen}, {x.shape[-1]})"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # fixed bugs for using kv_cahce
    freqs_cis = reshape_for_broadcast(freqs_cis, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, -xq.size(1):]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.head_dim = model_args.head_dim
        self.n_kv_heads = model_args.n_kv_heads
        self.initializer_range = model_args.initializer_range
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)
        self.use_qk_norm = model_args.qk_norm
        self.qk_norm_after_rope = model_args.qk_norm_after_rope

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    def init_weights(self, range_scale: float):
        if self.use_qk_norm:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.wo.weight, mean=0.0, std=self.initializer_range * range_scale)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache: AttentionKVCache | None):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv)
            is_decoding = xq.size(1) <= 1
            xk = xk.to(xq.device)
            xv = xv.to(xq.device)
        else:
            is_decoding = False

        # Apply QK normalization and RoPE based on configuration order
        if self.qk_norm_after_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            if self.use_qk_norm:
                xq = self.q_norm(xq)
                xk = self.k_norm(xk)
        else:
            if self.use_qk_norm:
                xq = self.q_norm(xq)
                xk = self.k_norm(xk)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        if is_decoding:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=False)
        else:
            output = self.sdpa(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.hidden_dim = model_args.ffn_hidden_dim
        self.initializer_range = model_args.initializer_range

        self.w1 = nn.Linear(model_args.dim, model_args.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(model_args.ffn_hidden_dim, model_args.dim, bias=False)
        self.w3 = nn.Linear(model_args.dim, model_args.ffn_hidden_dim, bias=False)

    def init_weights(self, range_scale: float):
        nn.init.normal_(self.w1.weight, mean=0.0, std=self.initializer_range)
        for linear in (self.w2, self.w3):
            nn.init.normal_(linear.weight, mean=0.0, std=self.initializer_range * range_scale)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AttentionBlock(nn.Module):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        # NOTE: add _no_weight_decay attribute to all norm layers
        self.attention_norm._no_weight_decay = True
        self.moe_enabled = False

    def init_weights(self, init_range_scale: float):
        self.attention_norm.reset_parameters()
        self.attention.init_weights(init_range_scale)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        seq_idx: torch.Tensor | None,
        kv_cache: AttentionKVCache | None,
        *args
    ):
        out = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache=kv_cache)
        return out


class FeedForwardBlock(nn.Module):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()
        self.feed_forward = FeedForward(model_args)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        # NOTE: add _no_weight_decay attribute to all norm layers
        self.ffn_norm._no_weight_decay = True
        self.moe_enabled = False

    def init_weights(self, init_range_scale: float):
        self.ffn_norm.reset_parameters()
        self.feed_forward.init_weights(init_range_scale)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None, seq_idx: torch.Tensor | None, *args):
        out = x + self.feed_forward(self.ffn_norm(x))
        return out


_moe_metrics_hook = MetricsAvgAccumulatorHook()


class MoEBlock(nn.Module):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()
        self.num_experts = model_args.num_experts
        self.initializer_range = model_args.initializer_range
        self.load_balance_coeff = model_args.load_balance_coeff

        if model_args.use_heterogeneous_moe:
            self.moe = HeterogeneousMoE(model_args)
        else:
            self.moe = MoE(model_args)

        self.moe_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        # NOTE: add _no_weight_decay attribute to all norm layers
        self.moe_norm._no_weight_decay = True
        self.moe_enabled = True
        self.enable_log_expert_bias = model_args.enable_log_expert_bias

    def init_weights(self, init_range_scale: float):
        self.moe_norm.reset_parameters()
        self.moe.init_weights(self.initializer_range * init_range_scale, self.moe_norm.weight.device)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None, seq_idx: torch.Tensor | None, *args):
        out = x + self.moe(self.moe_norm(x))
        if self.load_balance_coeff is not None and self.enable_log_expert_bias:
            expert_bias_metrics = {
                f"expert-bias/{i}": self.moe.expert_bias.detach()[i] for i in range(self.num_experts)
            }
            std, mean = torch.std_mean(self.moe.expert_bias.detach())
            expert_bias_metrics["expert-bias/mean"] = mean
            expert_bias_metrics["expert-bias/std"] = std
            push_extra_metrics(_moe_metrics_hook, **expert_bias_metrics)
        return out


class Hybrid(nn.Module, ModelProtocol):

    def __init__(self, model_args: HybridModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        # NOTE: add _no_weight_decay attribute to embedding and output layers
        self.tok_embeddings._no_weight_decay = True

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        logger.info(f"Hybrid pattern: {model_args.hybrid_patterns}")
        self.hybrid_pattern = model_args.hybrid_patterns

        self.layers = torch.nn.ModuleDict()
        for layer_idx, pattern in enumerate(self.hybrid_pattern):
            match pattern:
                case "f":
                    self.layers[str(layer_idx)] = FeedForwardBlock(model_args)
                case "a":
                    self.layers[str(layer_idx)] = AttentionBlock(model_args)
                case "e":
                    self.layers[str(layer_idx)] = MoEBlock(model_args)
                case _:
                    raise ValueError("Unsupported pattern: {pattern}.")

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        # NOTE: add _no_weight_decay attribute to all norm layers
        self.norm._no_weight_decay = True

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # NOTE: add _no_weight_decay attribute to embedding and output layers
        self.output._no_weight_decay = True
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=self.model_args.initializer_range)
        # init_range_scale = 1 / math.sqrt(2 * self.model_args.n_layers / 2)
        init_range_scale = 1.0
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_range_scale)
        if self.norm is not None:
            self.norm.reset_parameters()
        if self.output is not None:
            nn.init.normal_(
                self.output.weight,
                std=self.model_args.initializer_range
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.head_dim,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        eos_id: int | None = None,  # TODO: temp align with titan
        input_batch: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        cache_params: List[InferenceCache | None] | None = None
    ):
        # TODO: fix this in the future
        self.freqs_cis = self._precompute_freqs_cis().to(tokens.device)

        batch = input_batch if input_batch is not None else tokens

        # calculate seq idx for var len
        if self.model_args.attn_mask_type == "block_causal":
            new_seq_pos = F.pad((batch[:, :-1] == self.eos_id).to(torch.int32), (1, 0))
            seq_idx = torch.cumsum(new_seq_pos, dim=1).to(torch.int32)
        else:
            seq_idx = None

        if self.model_args.use_flex_attn:
            init_attention_mask(batch, eos_id=self.eos_id)

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if cache_params is None:
            cache_params = [None] * len(self.layers)

        for i, (layer, cache) in enumerate(zip(self.layers.values(), cache_params)):
            if getattr(layer, "moe_enabled", False):
                label = f"hmoe_moe_block/layer_{i}"
            elif getattr(layer, "attention", None) is not None:
                label = f"hmoe_attention_block/layer_{i}"
            else:
                label = f"hmoe_ffn_block/layer_{i}"
            with record_function(label):
                h = layer(h, self.freqs_cis, seq_idx, cache)

        h = self.norm(h) if self.norm else h
        if return_hidden_states:
            return h
        output = self.output(h) if self.output else h
        return output

    def alloc_inference_caches(self,
                               max_seq_len: int | None = None,
                               kvcache_cpu_offload: bool = False
                               ) -> List[InferenceCache | None]:

        if max_seq_len is None:
            max_seq_len = self.model_args.max_seq_len

        cache_params = []
        for pattern in self.hybrid_pattern:
            match pattern:
                case "f":
                    cache_params.append(None)
                case "a":
                    cache_params.append(AttentionKVCache(max_seq_len, cpu_offload=kvcache_cpu_offload))
                case "e":
                    cache_params.append(None)
                case _:
                    raise ValueError("Unsupported pattern: {pattern}.")
        return cache_params
