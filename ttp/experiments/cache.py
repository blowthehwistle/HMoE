import torch
from typing import Tuple


class InferenceCache:

    def __init__(self):
        self.reset()

    @property
    def inited(self) -> bool:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class AttentionKVCache(InferenceCache):

    def __init__(self, max_seq_len: int, cpu_offload: bool = False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cpu_offload = cpu_offload

    @property
    def inited(self) -> bool:
        return self.k_cache is not None

    def reset(self):
        self.pos = 0
        self.k_cache = None
        self.v_cache = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cpu_offload:
            # TODO optimize it
            k = k.cpu()
            v = v.cpu()
        bsz, slen, n_kv_heads_local, head_dim = k.shape

        if not self.inited:
            self.k_cache = torch.empty((bsz, self.max_seq_len, n_kv_heads_local, head_dim), device=k.device, dtype=k.dtype)
            self.v_cache = torch.empty((bsz, self.max_seq_len, n_kv_heads_local, head_dim), device=v.device, dtype=v.dtype)

        assert self.pos + slen <= self.max_seq_len, \
            f"KV cache overflow: pos={self.pos}, slen={slen}, max_len={self.max_seq_len}"

        # Concatenate the new keys and values to the existing cache
        self.k_cache[:, self.pos:self.pos + slen].copy_(k)
        self.v_cache[:, self.pos:self.pos + slen].copy_(v)

        # Update the position
        self.pos += slen

        # Return the updated cache
        return self.k_cache[:bsz, :self.pos], self.v_cache[:bsz, :self.pos]
