from huggingface_hub import snapshot_download

num_b_tokens = 10 # 10BT, 100BT, 350BT

print(f"Downloading HuggingFaceFW/fineweb-edu/sample/{num_b_tokens}BT dataset...")

snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    allow_patterns=f"sample/{num_b_tokens}BT/*.parquet",
    local_dir=f"fineweb_{num_b_tokens}BT",
    local_dir_use_symlinks=False,
    resume_download=True,
)