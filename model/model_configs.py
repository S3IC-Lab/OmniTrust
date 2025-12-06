import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_configs = {
    # ======= openai ========
    "chatgpt": os.path.join(project_root, "config/configs/open_ai_chat_example.yaml"),
    "gpt4": os.path.join(project_root, "config/configs/open_ai_gpt4_example.yaml"),
    "gpt-3.5-turbo-16k": os.path.join(project_root, "config/configs/open_ai_gpt_3_5_16k.yaml"),
    "instructgpt": os.path.join(project_root, "config/configs/open_ai_completion_example.yaml"),
    "openassist": os.path.join(project_root, "config/configs/open_assist_llm_example.yml"),
    # ======= huggingface management ========
    "Qwen2-7B-Instruct": os.path.join(project_root, "config/configs/qwen2_llm_7b.yml"),
    # =======  ========
    "cohere": os.path.join(project_root, "config/configs/cohere_llm_example.yaml"),
    "alpaca": os.path.join(project_root, "config/configs/alpaca_lora_example.yml"),
    "baize": os.path.join(project_root, "config/configs/baize_llm_example.yml"),
    "dolly": os.path.join(project_root, "config/configs/dolly_llm_example.yml"),
    "koala": os.path.join(project_root, "config/configs/koala_llm_example.yml"),
    "mpt": os.path.join(project_root, "config/configs/mpt_llm_example.yml"),
    "redpajama": os.path.join(project_root, "config/configs/redpajama_llm_example.yml"),
    "vicuna": os.path.join(project_root, "config/configs/vicuna_llm_7b_example.yml"),
    "wizardlm": os.path.join(project_root, "config/configs/wizard_llm_example.yml"),
    # ======= ollama management ========
    "deepseek-r1:70b": os.path.join(project_root, "config/configs/deepseek_70b_example.yml"),
    "llama3.2:1b": os.path.join(project_root, "config/configs/llama3_2_example.yml"),
    "llama3.1:8b": os.path.join(project_root, "config/configs/llama3_1_8b.yml"),
    "llama3.3:70b": os.path.join(project_root, "config/configs/llama3_3_70b.yml"),
    "llama3:8b": os.path.join(project_root, "config/configs/llama3_8b.yml"),
    "shrijayan/llama-2-7b-chat-q2k:latest": os.path.join(project_root, "config/configs/llama-2-7b-chat-q2k.yml"),
    "llama2:7b": os.path.join(project_root, "config/configs/llama2_7b.yml"),    
    # ======= 自定义API端点 ========
    "DeepSeek-V3": os.path.join(project_root, "config/configs/deepseek_v3_example.yaml")
}

if __name__ == '__main__':
    print(model_configs)
