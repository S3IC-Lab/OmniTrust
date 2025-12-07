# AdvBench Dataset

This folder is intended to store the **AdvBench** dataset locally.

The dataset is hosted on Hugging Face:

🔗 https://huggingface.co/datasets/S3IC/AdvBench

## Download with Python

To download the dataset files (`advbench.csv` and `advbench.json`) into this folder:

```python
from huggingface_hub import hf_hub_download

repo = "S3IC/AdvBench"

hf_hub_download(repo_id=repo, filename="advbench.csv", repo_type="dataset", local_dir=".")
hf_hub_download(repo_id=repo, filename="advbench.json", repo_type="dataset", local_dir=".")
```
