import functools
import torch.nn as nn
import torchtitan.components.optimizer
from torch.optim import Optimizer
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from ttp.patches import as_patch
from typing import Any, TypeVar


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(torchtitan.components.optimizer.OptimizersContainer):

    @as_patch
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            no_decay_params = []
            decay_params = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if hasattr(param, "_no_weight_decay"):
                        if param._no_weight_decay:
                            no_decay_params.append(param)
                        else:
                            decay_params.append(param)
                    else:
                        decay_params.append(param)

            params = [
                {"params": decay_params},
                {"params": no_decay_params, "weight_decay": 0.0}
            ]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=False),  # TODO flatten_optimizer_state_dict=True have bug
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=False),  # TODO flatten_optimizer_state_dict=True have bug
        )
        list(map(func, self.model_parts, self.optimizers))


def do_patch():
    torchtitan.components.optimizer.OptimizersContainer = OptimizersContainer
