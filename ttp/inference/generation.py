import torch
from typing import Optional


def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def get_ppl(
    model,
    input_ids: torch.Tensor,
):
    logits = model(input_ids)  # (B, T, vocab_size) assert B=1
    if type(logits) is not torch.Tensor:
        logits = logits.logits
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = input_ids[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(
        reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)).view(shift_labels.size())

    ce_loss = loss.mean(-1).cpu().detach().numpy()

    return ce_loss.tolist()


def get_cum_log_probs(
    model,
    tokens: torch.Tensor,
):

    logits = model(tokens)  # (B, T, vocab_size)
    if type(logits) is not torch.Tensor:
        logits = logits.logits
    gen_log_probs = torch.log_softmax(logits, dim=-1)
    gen_log_probs = gen_log_probs[:, :-1]
    labels = tokens[:, 1:]
    selected_log_probs = torch.gather(
        gen_log_probs, dim=-1, index=labels[..., None])[..., 0]
    cum_log_probs = torch.exp(selected_log_probs.mean(-1)).tolist()
    return cum_log_probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    cache_params=None,
) -> torch.Tensor:

    if cache_params is not None:
        logits = model(x, cache_params=cache_params)  # (B, T, vocab_size)
    else:
        logits = model(x)

    if type(logits) is not torch.Tensor:  # for huggingface model
        logits = logits.logits

    if logits.ndim == 4:
        # (num_mtp_heads, B, T, vocab_size) # for model with mtp
        logits = model(x)[0]

    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)

    next_token = multinomial_sample_one(probs, rng=rng)

    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    eos_id: Optional[int] = None,
    stopping_criteria: Optional[list] = None,
    use_kv_caches: bool = False,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    generated_tokens = input_ids.clone()

    bsz, slen = input_ids.shape

    if use_kv_caches:
        cache_params = model.alloc_inference_caches()
    else:
        cache_params = None

    for gen_token_id in range(max_new_tokens):

        if gen_token_id > 0 and use_kv_caches:
            for_generated_tokens = generated_tokens[:, -1:]  # last token
        else:
            for_generated_tokens = generated_tokens  # input_ids

        next_token = generate_next_token(
            model,
            x=for_generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
            cache_params=cache_params,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        if eos_id is not None and input_ids.shape[0] == 1 and (
                next_token == eos_id).any():
            return generated_tokens

        if input_ids.shape[0] == 1 and stopping_criteria:
            for stop_ids in stopping_criteria:
                if generated_tokens.shape[1] >= len(stop_ids):
                    if generated_tokens[0, -
                                        len(stop_ids):].tolist() == stop_ids:
                        return generated_tokens

    return generated_tokens
