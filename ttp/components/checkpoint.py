import os
import re
import traceback
import torch.distributed as dist
import torchtitan.components.checkpoint
from torchtitan.components.checkpoint import MODEL, DATALOADER, AsyncMode
from torchtitan.tools.logging import logger
from ttp.patches import as_patch
from typing import Any


class CheckpointManager(torchtitan.components.checkpoint.CheckpointManager):

    @as_patch
    def __init__(self, *args, **kwargs) -> None:
        # Use args, kwargs to make sure if official torchtitan changes the arguments, patches are still compatible
        super().__init__(*args, **kwargs)

    @as_patch
    def load(self, step=-1):
        try:
            return super().load(step)
        except:  # noqa: E722 FIX ray log bug
            raise RuntimeError(traceback.print_exc())

    @as_patch
    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        # For the first step, we will only load the model weights.
        if model_only:
            # NOTE: temp add MODEL key for compatible with the past
            model_state_dict = self.states[MODEL].state_dict()

            # NOTE: temp rm expert_bias, tokens_per_expert from model_state_dict
            filtered_out_parameter_names = [
                k for k in model_state_dict.keys() if 'expert_bias' in k or 'tokens_per_expert' in k or 'freqs_cis' in k]
            filtered_state_dict = {k: v for k, v in model_state_dict.items() if k not in filtered_out_parameter_names}
            logger.info(f"Filtered out parameter names: {filtered_out_parameter_names}")

            return {MODEL: filtered_state_dict}

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        states_to_load = self._flattened_model_states_sd(states_to_load)

        if self.ft_manager:
            states_to_load.pop(DATALOADER)

        return states_to_load

    @as_patch
    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.
        if os.environ.get("TTP_SKIP_FINAL_CHECKPOINT") == "1":
            logger.info(f"Skipping the final checkpoint at step {curr_step}.")
            return

        logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
        states = self._flattened_model_states_sd()

        if self.last_save_in_hf:
            raise ValueError("Saving in HF format is not supported.")

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )

    @as_patch
    def _purge_stale_checkpoints(self):
        """
        This patch is to fix unnecessary error reports caused by non-ckpt files in the ckpt folder.
        """
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
            and (not self.ft_manager or self.ft_manager.participating_rank() == 0)
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                if filename.startswith("step-"):
                    match = re.search(r"step-(\d+)", filename)
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)


def do_patch():
    torchtitan.components.checkpoint.CheckpointManager = CheckpointManager
