#!/usr/bin/env python3
"""Download selected RedPajama-Data-1T shards from the official URL manifests.

The Hugging Face dataset repo stores URL manifests, while the actual JSONL shards
live on Together's CDN. This script keeps the original directory layout under
``redpajama_1T`` so Hugging Face ``load_dataset("json", data_files=...)`` can
read the downloaded shards locally.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download


REPO_ID = "togethercomputer/RedPajama-Data-1T"
BASE_URL_PREFIX = "https://data.together.xyz/redpajama-data-1T/"
DEFAULT_SUBSETS = ["c4", "github", "arxiv"]
SUBSET_MANIFESTS = {
    "common_crawl": "urls/common_crawl.txt",
    "c4": "urls/c4.txt",
    "github": "urls/github.txt",
    "arxiv": "urls/arxiv.txt",
    "wikipedia": "urls/wikipedia.txt",
    "stackexchange": "urls/stackexchange.txt",
}


def read_manifest(subset: str) -> list[str]:
    manifest = SUBSET_MANIFESTS[subset]
    path = hf_hub_download(REPO_ID, manifest, repo_type="dataset", token=True)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def url_to_local_path(url: str, output_dir: Path) -> Path:
    if url.startswith(BASE_URL_PREFIX):
        rel = url[len(BASE_URL_PREFIX):]
    else:
        rel = urlparse(url).path.lstrip("/")
    return output_dir / rel


def download_url(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"skip existing {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"download {url}", flush=True)
    print(f"    -> {dest}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
    tmp.rename(dest)
    print(f"done {dest} ({dest.stat().st_size / (1024 ** 3):.2f} GiB)", flush=True)


def write_selected_manifest(urls: list[str], output_dir: Path) -> None:
    manifest = output_dir / "selected_urls.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        for url in urls:
            f.write(url + "\n")
    print(f"wrote {manifest}", flush=True)


def completed_data_bytes(output_dir: Path) -> int:
    data_root = output_dir / "v1.0.0"
    if not data_root.exists():
        return 0
    total = 0
    for path in data_root.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            total += path.stat().st_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("redpajama_1T"))
    parser.add_argument("--subsets", nargs="+", choices=sorted(SUBSET_MANIFESTS), default=DEFAULT_SUBSETS)
    parser.add_argument("--max-files-per-subset", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Download all files for the selected subsets")
    parser.add_argument(
        "--max-gb",
        type=float,
        default=None,
        help="Stop after completed data files under output-dir reach this many GiB.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.all and args.max_files_per_subset <= 0:
        print("--max-files-per-subset must be positive", file=sys.stderr)
        raise SystemExit(2)

    selected_urls: list[str] = []
    for subset in args.subsets:
        urls = read_manifest(subset)
        if args.all:
            selected = urls[args.start_index:]
        else:
            selected = urls[args.start_index: args.start_index + args.max_files_per_subset]
        if not selected:
            print(f"no URLs selected for subset={subset}", file=sys.stderr)
            continue
        print(f"subset={subset}: selecting {len(selected)} of {len(urls)} URLs", flush=True)
        selected_urls.extend(selected)

    write_selected_manifest(selected_urls, args.output_dir)

    max_bytes = None if args.max_gb is None else int(args.max_gb * (1024 ** 3))
    for url in selected_urls:
        if max_bytes is not None:
            current_bytes = completed_data_bytes(args.output_dir)
            if current_bytes >= max_bytes:
                print(
                    f"Reached --max-gb={args.max_gb:.2f} GiB "
                    f"({current_bytes / (1024 ** 3):.2f} GiB complete). Stopping.",
                    flush=True,
                )
                break
        download_url(url, url_to_local_path(url, args.output_dir), args.force)


if __name__ == "__main__":
    main()
