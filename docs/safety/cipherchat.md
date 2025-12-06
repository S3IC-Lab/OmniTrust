
***
# CipherChat

**Dataset:** safetyprompts (Chinese safety assessment benchmark)  
**Attack Category:** Jailbreak / Adversarial Prompt Encoding  
**Target Models:**  
- gpt-3.5-turbo-0613  
- gpt-4-0613  
- text-davinci-003  
- text-curie-001  
- text-babbage-001  


---

# 🔍 Overview

CipherChat is a jailbreak method that encodes harmful instructions using various cipher transformations (e.g., Morse code, substitution encoding) to bypass model safety filters. The LLM is then guided to decode and respond to the original harmful request.

---

# 🧩 Attack Parameters

| Parameter | Description |
|----------|-------------|
| `--attack CipherChat` | Selects CipherChat attack method |
| `--dataset` | Dataset name (e.g., `safetyprompts`) |
| `--data-dir` | Path to dataset directory |
| `--output-file` | Save path for attack results |
| `--model-path` | Path or name of target model |
| `--model_name` | Model name string (`gpt-3.5-turbo-0613`, default) |
| `--encode_method` | Encoding type (`morse`, `unchange`, etc.) |
| `--instruction_type` | Harmful domain category (default: `Crimes_And_Illegal_Activities`) |
| `--demonstration_toxicity` | Toxicity label (`toxic`, default) |
| `--language` | Prompt language (`zh`, default) |

---

# 🔧 Environment Setup

change it into real ones...

# 🚀 Run CipherChat

**Example command:**

```bash
  python examples/jailbreak.py --attack CipherChat --dataset safetyprompts --data-dir ./data/dataset/safetyprompts --output-file attack/jailbreak/utils/CipherChat/saved_results --model-path /home/balei/models/vicuna/vicuna-13b-v1.5/ CipherChat --model_name gpt-3.5-turbo-0613 --encode_method morse --instruction_type Crimes_And_Illegal_Activities --demonstration_toxicity toxic --language zh
```

Note: The second `CipherChat` is used to indicate the subparser `cipherchat_parser = subparsers.add_parser("CipherChat", help="Run CipherChat attack")`.

# 📁 Output

**Saved Results**: `attack/jailbreak/utils/CipherChat/saved_results` 

* ![image-20250302172256937](assets/image-20250302172256937.png)
* ![image-20250302172600383](assets/image-20250302172600383.png)

# ✔ Summary

CipherChat is a practical and modular jailbreak attack that evaluates an LLM’s ability to resist harmful encoded instructions. It supports multiple ciphers, toxic domains, and multilingual prompts.
