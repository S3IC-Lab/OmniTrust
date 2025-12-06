## Local LLM deployment - take deepseek-llm-7b-chat as an example

### Step 1: Download the local model (ignore this step if you already have a local model )

The document uses `deepseek-llm-7b-chat`(https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) as an example

```bash
huggingface-cli download --resume-download deepseek-ai/deepseek-llm-7b-chat --local-dir ~/model/deepseek-llm-7b-chat
```

**NOTE**: you can use this command if you cannot connect the huggingface in your area.
```shell
export HF_ENDPOINT=https://hf-mirror.com
```

The local model address is：`/home/xx_x/model/deepseek-llm-7b-chat`

### Step 2: Implement the model adapter (already done at infrastructure time)

Implement a model adapter for the new model in `/model/model_adapter.py`. You can follow the existing example.

such as，`deepseekchatadapter`（already have）

```python
class DeepseekChatAdapter(BaseModelAdapter):
    """The model adapter for deepseek-ai's chat models"""

    # Note: that this model will require tokenizer version >= 0.13.3 because the tokenizer class is LlamaTokenizerFast

    def match(self, model_path: str):
        return "deepseek-llm" in model_path.lower() and "chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("deepseek-chat")
```


And add a new example using `register_model_adapter`.

```python
register_model_adapter(DeepseekChatAdapter)
```

### Step 3: Modify the model registration information (need to add)

Write more information to the supported models in `model_registry.py`

For example: add `deepseek-llm-7b-chat` information

```python
register_model_info(
    ["deepseek-llm-67b-chat",
     "deepseek-llm-7b-chat",
    ],
    "DeepSeek LLM",
    "https://huggingface.co/deepseek-ai/",
    "An advanced language model by DeepSeek",
)
```

## LVLMs
Scripts to load local multimodal LLMs are listed in folder `lvlm`

## Ollama, OpenAI Call
`llm.py` provide the full pipeline to call models by ollama and openai interfaces. 
`model_configs.py` provide config callers, which provided in `~/config/configs`

## API Configuration
For models to call with API, please specify the API and base URL in `~/config/api_config.yaml`
Get the api and url by below code:
```python
from config import GET_API
api, url = GET_API("your model name")
```