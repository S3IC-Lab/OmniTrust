# Humaneval Dataset

This folder is intended to store the **Humaneval** dataset locally.

The dataset is hosted on Hugging Face:

🔗https://huggingface.co/datasets/S3IC/humaneval

## Download with Python

To download the dataset files (`humaneval.jsonl`) into this folder:

```python
from huggingface_hub import hf_hub_download

repo = "S3IC/humaneval"

hf_hub_download(repo_id=repo, filename="humaneval.jsonl", repo_type="dataset", local_dir=".")
```