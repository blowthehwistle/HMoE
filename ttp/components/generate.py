import torch


def multinomial_sample_one(probs: torch.Tensor, rng: torch.Generator | None) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    last_token_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: torch.Generator | None = None
) -> torch.Tensor:
    probs = logits_to_probs(last_token_logits, temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token
