# C4 Dataset

This folder is intended to store the **C4** dataset locally.

The dataset is hosted on Hugging Face:

🔗 https://huggingface.co/datasets/S3IC/c4

## Download with Python

To download the dataset files (`c4.json`) into this folder:

```python
from huggingface_hub import hf_hub_download

repo = "S3IC/c4"

hf_hub_download(repo_id=repo, filename="c4.json", repo_type="dataset", local_dir=".")
```