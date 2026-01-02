# CNN_Dailymail Dataset

This folder is intended to store the **CNN_Dailymail** dataset locally.

The dataset is hosted on Hugging Face:

🔗 https://huggingface.co/datasets/S3IC/cnn_dailymail

## Download with Python

To download the dataset files (`cnn_dailymail.jsonl`) into this folder:

```python
from huggingface_hub import hf_hub_download

repo = "S3IC/cnn_dailymail"

hf_hub_download(repo_id=repo, filename="cnn_dailymail.jsonl", repo_type="dataset", local_dir=".")
```