import os
import sys
import torch
import logging
from typing import Optional
from torchtitan.train import Trainer
from torchtitan.config.manager import ConfigManager
from torchtitan.distributed import utils as dist_utils
from ttp.patches import register_models, as_patch
from ttp.config.job_config import TTPJobConfig
from ttp.components.metrics import push_extra_metrics, MetricsAvgAccumulatorHook
from ttp.utils.nvtx import nvtx_range


_lm_loss_hook = MetricsAvgAccumulatorHook(need_accum=False)
_aux_load_balance_loss_hook = MetricsAvgAccumulatorHook()
_aux_p_penalty_loss_hook = MetricsAvgAccumulatorHook()
_moe_imbalance_hook = MetricsAvgAccumulatorHook()


class TTPTrainer(Trainer):

    def __init__(self, job_config: TTPJobConfig):
        super().__init__(job_config)
        self.metrics_processor.set_trainer(self)
        if os.path.isfile("/dockerdata/.tccl/tccl.data") and torch.distributed.get_rank() == 0:
            with open("/dockerdata/.tccl/tccl.data", 'a') as f:
                f.write(f"'--tensor-model-parallel-size', '{job_config.parallelism.tensor_parallel_degree}'\n")
                f.write(f"'--pipeline-model-parallel-size', '{job_config.parallelism.pipeline_parallel_degree}'\n")
                f.write(f"'--context-parallel-size', '{job_config.parallelism.context_parallel_degree}'\n")

    @as_patch
    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:

        # NOTE: broadcast flag to all ranks for debug
        if torch.cuda.is_available():
            flag = torch.tensor([], dtype=torch.float32, device="cuda")
            torch.distributed.broadcast(flag, 0, async_op=True)

        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["input"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )

        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    with nvtx_range("train/forward"):
                        pred = model_parts[0](inputs, eos_id=self.tokenizer.eos_id)
                    with nvtx_range("train/loss"):
                        loss = self.loss_fn(pred, labels)

                        # NOTE: Add auxiliary MoE load-balancing loss from all MoE layers, if enabled
                        loss = self._apply_aux_loss(model_parts, loss)

                        # NOTE: Record MoE imbalance statistics to tensorboard
                        self._record_moe_imbalance(model_parts)

                # need to free to before bwd to avoid peaking memory
                del pred
                with nvtx_range("train/backward"):
                    loss.backward()

        return loss

    def _apply_aux_loss(self, model_parts: list[torch.nn.Module], loss: torch.Tensor) -> torch.Tensor:
        load_balance_loss = None
        p_penalty_loss = None

        for module in model_parts[0].modules():
            # Handle auxiliary load balancing loss
            if hasattr(module, "last_load_balancing_loss") and hasattr(
                module, "aux_load_balance_loss_coeff"
            ):
                coeff = getattr(module, "aux_load_balance_loss_coeff", 0.0)
                if coeff is not None and float(coeff) > 0.0:
                    term = coeff * module.last_load_balancing_loss
                    load_balance_loss = term if load_balance_loss is None else load_balance_loss + term

            # Handle P-Penalty loss for heterogeneous MoE
            if hasattr(module, "last_p_penalty_loss") and hasattr(
                module, "p_penalty_coeff"
            ):
                coeff = getattr(module, "p_penalty_coeff", 0.0)
                if coeff is not None and float(coeff) > 0.0:
                    term = coeff * module.last_p_penalty_loss
                    p_penalty_loss = term if p_penalty_loss is None else p_penalty_loss + term

        metrics = {"loss/lm_loss": loss.detach()}
        push_extra_metrics(_lm_loss_hook, **metrics)

        if load_balance_loss is not None:
            metrics = {"loss/aux_load_balance_loss": load_balance_loss.detach()}
            push_extra_metrics(_aux_load_balance_loss_hook, **metrics)
            # Add load balance loss to the total loss
            loss = loss + load_balance_loss / self.gradient_accumulation_steps

        if p_penalty_loss is not None:
            metrics = {"loss/aux_p_penalty_loss": p_penalty_loss.detach()}
            push_extra_metrics(_aux_p_penalty_loss_hook, **metrics)
            # Add P-Penalty loss to the total loss
            loss = loss + p_penalty_loss / self.gradient_accumulation_steps

        return loss

    def _record_moe_imbalance(self, model_parts: list[torch.nn.Module]) -> None:
        moe_imbalance_metrics = {}

        for model_part in model_parts:
            for transformer_block in model_part.layers.values():
                if hasattr(transformer_block, "moe"):
                    moe = transformer_block.moe

                    tokens_per_expert = moe.tokens_per_expert
                    if tokens_per_expert is not None and tokens_per_expert.numel() > 0:
                        # Calculate imbalance coefficient: sum of absolute values of |mean - tokens_per_expert|
                        mean_tokens = tokens_per_expert.mean()
                        imbalance_coeffs = torch.abs(mean_tokens - tokens_per_expert)
                        total_imbalance = imbalance_coeffs.sum()

                        # Calculate relative imbalance (total imbalance / average token count)
                        relative_imbalance = total_imbalance / (mean_tokens + 1e-8)

                        # Get layer ID from transformer_block's parent
                        layer_id = None
                        for name, mod in model_part.named_modules():
                            if mod is transformer_block:
                                # Extract layer ID from full path, e.g., "layers.0" -> "0"
                                if "layers." in name:
                                    layer_id = name.split("layers.")[1]
                                else:
                                    layer_id = name
                                break

                        # Record to metrics
                        moe_imbalance_metrics.update({
                            f"moe_imbalance/layer_{layer_id}/relative_imbalance": relative_imbalance,
                        })

        # Record MoE imbalance statistics to tensorboard
        if moe_imbalance_metrics:
            push_extra_metrics(_moe_imbalance_hook, **moe_imbalance_metrics)


def main(logger: logging.Logger, args=sys.argv[1:]):
    register_models()
    logger.info(f"PyTorch Version: {torch.__version__}")
    config_manager = ConfigManager(TTPJobConfig)
    config: TTPJobConfig = config_manager.parse_args(args)

    # TODO pp bug
    if config.parallelism.pipeline_parallel_degree > 1 and config.training.global_batch_size > 0:
        raise ValueError("Pipeline parallel dosen't support global_batch_size > 0.")

    trainer: Optional[Trainer] = None
    try:
        if config.training.type_name == "ttp":
            trainer = TTPTrainer(config)
        elif config.training.type_name == "titan":
            trainer = Trainer(config)
        else:
            raise ValueError(f"Unsupported trainer type: {config.training.type_name}.")
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")


if __name__ == "__main__":
    from torchtitan.tools.logging import init_logger, logger

    init_logger()
    main(logger)
