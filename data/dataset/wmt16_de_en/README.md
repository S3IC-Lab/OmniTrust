# WMT16_DE_EN Dataset

This folder is intended to store the **WMT16_DE_EN** dataset locally.

The dataset is hosted on Hugging Face:

🔗 https://huggingface.co/datasets/S3IC/wmt16_de_en

## Download with Python

To download the dataset files (`wmt16_de_en.jsonl`) into this folder:

```python
from huggingface_hub import hf_hub_download

repo = "S3IC/wmt16_de_en"

hf_hub_download(repo_id=repo, filename="wmt16_de_en.jsonl", repo_type="dataset", local_dir=".")
```