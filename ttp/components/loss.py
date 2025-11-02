import torch
from ttp.config.job_config import TTPJobConfig
from torchtitan.tools.logging import logger
from ttp.experiments.kernels.loss.zero_loss import zero_loss
from ttp.experiments.kernels.loss.fused_cross_entropy import fused_cross_entropy


def _ignore_all(labels: torch.Tensor, ignore_index: int = -100) -> bool:
    return (labels != ignore_index).long().sum() == 0


def fused_cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if _ignore_all(labels):
        return zero_loss(pred)

    return fused_cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_fused_cross_entropy_loss(job_config: TTPJobConfig):
    if not job_config.parallelism.disable_loss_parallel:
        raise ValueError("TTP with fused_cross_entropy currently doesn't support loss_parallel.")
    return fused_cross_entropy_loss


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    if _ignore_all(labels):
        return zero_loss(pred)

    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_cross_entropy_loss(job_config: TTPJobConfig):
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def build_loss(job_config: TTPJobConfig):
    loss_function_type = job_config.training.loss_function_type
    logger.info(f"Building loss with type:{loss_function_type}.")
    match loss_function_type:
        case "cross_entropy":
            loss_fn = build_cross_entropy_loss(job_config)
        case "fused_cross_entropy":
            loss_fn = build_fused_cross_entropy_loss(job_config)
        case _:
            raise ValueError(f"Unsupported loss type: {loss_function_type}.")
    return loss_fn
