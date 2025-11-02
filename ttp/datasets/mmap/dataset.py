import os
import time
import math
import torch
import functools
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset
from torchtitan.tools.logging import logger
from ttp.tools.logging import timer
from ttp.datasets.mmap.indexed import make_indexed_dataset
from typing import List


class MMapDataset(Dataset):

    def __init__(
        self,
        name: str,
        data_prefix: str,
        num_samples: int | None,
        seq_length: int,
        shuffle_doc_idx: bool,
        shuffle_sample_idx: bool,
        seed: int,
        sft: bool,
        sft_loss_mask_token_minimum_id: int,
        loss_mask_token_ids: List[int] | None,
        sft_loss_mask_token_ids: List[int] | None
    ):
        self.name = name

        logger.info(f"Loading {name} dataset from {data_prefix}...")
        self.indexed_dataset = make_indexed_dataset(data_prefix)

        # Add for data monitor
        self.file_name = os.path.split(data_prefix)[1] + '.bin'
        # Build index mappings.

        logger.info(f"Building index mappings for {name} dataset from {data_prefix}...")
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            self.indexed_dataset.sizes,
            num_samples,
            seq_length,
            shuffle_doc_idx,
            shuffle_sample_idx,
            seed
        )
        logger.info(f"Loaded {name} dataset from {data_prefix} with {len(self)} samples OK!")

        # loss mask settings
        self.loss_mask_token_ids = loss_mask_token_ids

        # sft settings
        self.sft = sft
        self.sft_loss_mask_token_minimum_id = sft_loss_mask_token_minimum_id
        self.sft_loss_mask_token_ids = sft_loss_mask_token_ids

        self.loss_mask_list = []
        if self.loss_mask_token_ids is not None:
            self.loss_mask_list += self.loss_mask_token_ids
        if self.sft_loss_mask_token_ids is not None and self.sft:
            self.loss_mask_list += self.sft_loss_mask_token_ids

        if self.sft:
            logger.info(
                f"SFT Dataset: sft_loss_mask_token_minimum_id={sft_loss_mask_token_minimum_id} "
                f"sft_loss_mask_token_ids={sft_loss_mask_token_ids}"
            )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = idx % len(self.shuffle_idx)
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f],
                offset=offset_f,
                length=offset_l - offset_f + 1
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1))
            sample = np.concatenate(sample_list)
        tokens_ = torch.from_numpy(np.array(sample, dtype=np.int64)).long()

        # loss mask
        loss_mask = torch.ones_like(tokens_)
        if self.loss_mask_list:
            for mask_token in self.loss_mask_list:
                loss_mask[tokens_ == mask_token] = 0

        if self.sft:
            # a special number, i.e., 300000 is added to the tokens to mark which tokens should be masked and not calculate loss
            loss_mask[tokens_ >= self.sft_loss_mask_token_minimum_id] = 0
            tokens_ = tokens_ % self.sft_loss_mask_token_minimum_id

        if loss_mask.sum().item() == 0:
            raise ValueError("No tokens to calculate loss, check data and tokenizer!")

        input_ids = tokens_[:-1].contiguous()
        labels = tokens_[1:].clone()
        labels[loss_mask[1:] == 0] = -100
        return input_ids, labels


def _build_index_mappings(
    name,
    data_prefix,
    sizes,
    num_samples,
    seq_length,
    shuffle_doc_idx,
    shuffle_sample_idx,
    seed
):
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = sizes.sum()
    total_num_of_documents = sizes.shape[0]
    logger.info(f"Buiding Index Mappings: {tokens_per_epoch} tokens/epoch\t {total_num_of_documents} documents.")
    num_epochs, num_samples = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    num_samples_first_epoch = math.ceil((tokens_per_epoch - 1) / seq_length)
    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    while not all([
        os.path.isfile(doc_idx_filename),
        os.path.isfile(sample_idx_filename),
        os.path.isfile(shuffle_idx_filename)
    ]):
        logger.info(f"MMap Cache: {doc_idx_filename} isfile={os.path.isfile(doc_idx_filename)}")
        logger.info(f"MMap Cache: {sample_idx_filename} isfile={os.path.isfile(sample_idx_filename)}")
        logger.info(f"MMap Cache: {shuffle_idx_filename} isfile={os.path.isfile(shuffle_idx_filename)}")
        if torch.distributed.get_rank() == 0:
            logger.warning(f"Could not find index map files ({_filename}*.npy), building the indices on rank 0 ...")
            np_rng = np.random.default_rng(seed=seed)
            with timer(" > elasped time to build and save doc-idx mapping: {delta}"):
                doc_idx = _build_doc_idx(num_epochs, total_num_of_documents, shuffle_doc_idx, np_rng)
                _save_and_check(doc_idx_filename, doc_idx, allow_pickle=True)
            # sample-idx.
            with timer(" > elasped time to build and save sample-idx mapping: {delta}"):
                sample_idx = _build_sample_idx(sizes, doc_idx, seq_length, num_samples)
                assert sample_idx.shape[0] - 1 == num_samples
                _save_and_check(sample_idx_filename, sample_idx, allow_pickle=True)
            # shuffle-idx.
            with timer(" > elasped time to build and save shuffle-idx mapping: {delta}"):
                shuffle_idx = _build_shuffle_idx(num_samples, num_samples_first_epoch, shuffle_sample_idx, np_rng)
                _save_and_check(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
        else:
            time.sleep(5)

    logger.info(" > waiting for all ranks to finish building the index ")
    with timer(" > all ranks finished building the index: {delta}"):
        dist.barrier()

    # Load mappings.
    with timer("    loaded indexed file in {delta}"):
        with timer(" > loading doc-idx mapping from {filename}: {delta}", filename=doc_idx_filename):
            doc_idx = _load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
        with timer(" > loading sample-idx mapping from {filename}: {delta}", filename=sample_idx_filename):
            sample_idx = _load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
        with timer(" > loading shuffle-idx mapping from {filename}: {delta}", filename=shuffle_idx_filename):
            shuffle_idx = _load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    logger.info('    total number of samples: {}'.format(len(sample_idx)))
    logger.info('    total number of epochs: {}'.format(num_epochs))
    return doc_idx, sample_idx, shuffle_idx


def _retry(max_retries: int, delay: int):
    def _(func):
        @functools.wraps(func)
        def _rfunc(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    logger.info(f"> call {func.__name__} [{retries + 1}/{max_retries}]")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.exception(e)
                    retries += 1
                    time.sleep(delay)
            raise RuntimeError(f"Max retry call {func.__name__}: {max_retries}.")
        return _rfunc
    return _


@_retry(3, 5)
def _save_and_check(file_name, data, allow_pickle=True):
    np.save(file_name, data, allow_pickle=allow_pickle)
    np.load(file_name, allow_pickle=allow_pickle, mmap_mode='r')


@_retry(3, 5)
def _load(file_path, allow_pickle=True, mmap_mode='r'):
    return np.load(file_path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)


def _num_epochs(tokens_per_epoch, seq_length, num_samples: int | None = None):
    if num_samples is None:
        num_samples = (tokens_per_epoch - 1) // seq_length
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs, num_samples


def _build_doc_idx(num_epochs, total_num_of_documents, shuffle_doc_idx, rng):
    doc_idx = np.mgrid[:num_epochs, :total_num_of_documents][1]
    if shuffle_doc_idx:
        doc_idx = rng.permuted(doc_idx, axis=1)
    return doc_idx.flatten()


def _build_sample_idx(sizes, doc_idx, seq_length, num_samples):
    # Total number of samples. For -1 see comments in `_num_epochs`.
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int64)

    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[0][0] = doc_idx_index
    sample_idx[0][1] = doc_offset
    for sample_index in tqdm(
        range(1, num_samples + 1),
        desc="Building sample idx...",
        total=num_samples,
        unit="sample"
    ):
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
    return sample_idx


def _build_shuffle_idx(num_samples, num_samples_first_epoch, shuffle_sample_idx, rng):
    shuffle_idx = np.arange(num_samples, dtype=np.int64)
    if shuffle_sample_idx:
        logger.info(" > building shuffle_sample_idx for not_shuffle_doc_idx ...")
        # shuffle first epoch
        shuffle_idx[:num_samples_first_epoch] = rng.permuted(shuffle_idx[:num_samples_first_epoch])
        # shuffle other epoch
        shuffle_idx[num_samples_first_epoch:] = rng.permuted(shuffle_idx[num_samples_first_epoch:])
    return shuffle_idx


def load_mmap_dataset(
    data_name: str,
    data_prefix: str,
    num_samples: int | None,
    seq_length: int,
    shuffle_doc_idx: bool,
    shuffle_sample_idx: bool,
    seed: int,
    sft: bool,
    sft_loss_mask_token_minimum_id: int,
    loss_mask_token_ids: List[int] | None,
    sft_loss_mask_token_ids: List[int] | None
) -> MMapDataset:
    return MMapDataset(
        name=data_name,
        data_prefix=data_prefix,
        num_samples=num_samples,
        seq_length=seq_length,
        shuffle_doc_idx=shuffle_doc_idx,
        shuffle_sample_idx=shuffle_sample_idx,
        seed=seed,
        sft=sft,
        sft_loss_mask_token_minimum_id=sft_loss_mask_token_minimum_id,
        loss_mask_token_ids=loss_mask_token_ids,
        sft_loss_mask_token_ids=sft_loss_mask_token_ids
    )
