import torch
from torch.utils.data import Sampler
from torchdata.stateful_dataloader.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.tools.logging import logger
from torchtitan.components.dataloader import BaseDataLoader
from typing import List, Any


class ParallelAwareBatchSampler(Sampler[List[int]], Stateful):

    def __init__(
        self,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        num_samples: int,
        batch_sample_start_offset: int
    ):
        super().__init__()
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.batch_size = batch_size
        assert num_samples >= self.batch_size_per_iter, \
            f"num_samples={num_samples}, batch_size_per_iter={self.batch_size_per_iter}"
        self.num_samples = num_samples // self.batch_size_per_iter * self.batch_size_per_iter
        self._batch_indices = torch.arange(dp_rank, self.num_samples, dp_world_size).reshape(-1, batch_size)
        self._batch_sample_start_offset = batch_sample_start_offset

        if self._batch_sample_start_offset < 0 or self._batch_sample_start_offset >= len(self._batch_indices):
            raise IndexError(
                f"Batch sample start idx ({self._batch_sample_start_offset}) out of range."
            )

    @property
    def batch_sample_start_offset(self) -> int:
        return self._batch_sample_start_offset

    @property
    def batch_size_per_iter(self) -> int:
        return self.dp_world_size * self.batch_size

    def increase_batch_sample_idx(self):
        self._batch_sample_idx += 1

    def __len__(self) -> int:
        return len(self._batch_indices)

    def __iter__(self):
        for batch in self._batch_indices[self._batch_sample_start_offset:]:
            yield batch.tolist()

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank": self.dp_rank,
            "batch_size_per_iter": self.batch_size_per_iter,
            "batch_sample_start_offset": self._batch_sample_start_offset
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        old_batch_size_per_iter = state_dict["batch_size_per_iter"]
        batch_sample_start_offset = state_dict["batch_sample_start_offset"]

        consumed_samples = batch_sample_start_offset * old_batch_size_per_iter
        logger.info(f"Consumed samples = ({consumed_samples} = {old_batch_size_per_iter} * {batch_sample_start_offset}).")

        if consumed_samples % self.batch_size_per_iter != 0:
            logger.warning(
                "The consumed data cannot be evenly divided by the current batch size per iter "
                f"({self.batch_size_per_iter} = {self.dp_world_size} * {self.batch_size})."
            )
        self._batch_sample_start_offset = consumed_samples // self.batch_size_per_iter


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):

    def __init__(
        self,
        dataset,
        batch_sampler_offset: int,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        num_workers: int = 0
    ):
        self._batch_sample_idx = 0
        super().__init__(
            dataset,
            batch_sampler=ParallelAwareBatchSampler(
                dp_rank,
                dp_world_size,
                batch_size,
                num_samples=len(dataset),
                batch_sample_start_offset=self._batch_sample_idx + batch_sampler_offset
            ),
            num_workers=num_workers,
        )

    def __iter__(self):
        for (inputs, labels) in super().__iter__():
            self._batch_sample_idx += 1
            yield {"input": inputs}, labels

    def state_dict(self):
        return {
            "batch_sample_idx": self._batch_sample_idx,
            "batch_sampler": self.batch_sampler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any] | None = None) -> None:
        # State being empty is valid.
        if state_dict is None:
            state_dict = {}
        self._batch_sample_idx = state_dict["batch_sample_idx"]
        state_dict["batch_sampler"]["batch_sample_start_offset"] = (
            self._batch_sample_idx + self.batch_sampler.batch_sample_start_offset
        )
        self.batch_sampler.load_state_dict(state_dict["batch_sampler"])
