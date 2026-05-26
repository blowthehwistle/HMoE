from datasets import load_dataset
from ttp.patches import as_patch
from torchtitan.datasets.hf_datasets import DATASETS, DatasetConfig
from typing import Any, Dict


def _parse_data_files(dataset_path: str):
    if "," not in dataset_path:
        return dataset_path
    return [path.strip() for path in dataset_path.split(",") if path.strip()]


def _load_hf_dataset(dataset_path: str):
    return load_dataset(dataset_path, split='train', streaming=True)


def _load_json_dataset(dataset_path: str):
    """Load json dataset with default configuration."""
    return load_dataset('json', data_files=_parse_data_files(dataset_path), streaming=True, split='train')


def _load_parquet_dataset(dataset_path: str):
    """Load local parquet shards with default configuration."""
    return load_dataset('parquet', data_files=_parse_data_files(dataset_path), streaming=True, split='train')


def _process_hf_text(sample: Dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _process_banche_text(sample: Dict[str, Any]) -> str:
    """Process C4 dataset sample text."""

    keys = sample.keys()
    res = ''
    if 'title' in keys:
        if sample['title'] is not None:
            res = res + sample['title'] + "\n"

    if 'text' in keys:
        if sample['text'] is not None:
            res = res + sample['text']

    if 'input' in keys:
        if sample['input'] is not None:
            res = res + sample['input'] + "\n"

    if 'output' in keys:
        if sample['output'] is not None:
            res = res + sample['output']

    return res


@as_patch
def update_datasets():
    DATASETS.update({
        "hf": DatasetConfig(
            path=None,
            loader=_load_hf_dataset,
            text_processor=_process_hf_text,
        ),
        "json": DatasetConfig(
            path=None,
            loader=_load_json_dataset,
            text_processor=_process_hf_text,
        ),
        "parquet": DatasetConfig(
            path=None,
            loader=_load_parquet_dataset,
            text_processor=_process_hf_text,
        ),
        "banche": DatasetConfig(
            path=None,
            loader=_load_json_dataset,
            text_processor=_process_banche_text,
        ),
    })


def do_patch():
    update_datasets()
