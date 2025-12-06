import yaml
import os
from pathlib import Path

# ---------------------------------------------------------
# Helper: Replace ${ENV_VAR} with real environment variable
# ---------------------------------------------------------
def expand_env(value):
    """
    Replace ${ENV_VAR} patterns with actual environment values.
    Example:
        ${OPENAI_KEY} -> value of os.getenv("OPENAI_KEY")
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, None)
    return value


# ---------------------------------------------------------
# Load API configuration
# ---------------------------------------------------------
def LOAD_API_CONFIG(model_name=None):
    """
    Load api_config.yaml located in the same directory as this script.
    Automatically merges:
        default config   +   model-specific override
    """
    # Path to the YAML in the same folder as this .py
    config_path = Path(__file__).parent / "api_config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    api_cfg = cfg["api"]
    default_cfg = api_cfg["default"]
    models_cfg = api_cfg.get("models", {})

    # Select override config or fallback to empty
    model_cfg = models_cfg.get(model_name, {})

    # Merge: model_cfg overrides defaults
    merged = {**default_cfg, **model_cfg}

    # Replace environment variables
    merged = {k: expand_env(v) for k, v in merged.items()}

    # Store merged result inside cfg
    cfg["api"]["active"] = merged

    return cfg



# ---------------------------------------------------------
# NEW: Directly return api_key and api_url
# ---------------------------------------------------------
def GET_API(model_name=None):
    """
    Directly return:
        api_key, api_url
    Examples:
        key, url = get_api_credentials()
        key, url = get_api_credentials("deepseek-chat")
    """
    cfg = LOAD_API_CONFIG(model_name)
    active = cfg["api"]["active"]

    api_key = active.get("api_key")
    api_url = active.get("api_url")

    return api_key, api_url


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load default model
    cfg_default = LOAD_API_CONFIG()
    print(cfg_default["api"]["active"])

    # Load specific model
    cfg_ds = LOAD_API_CONFIG("deepseek-chat")
    print(cfg_ds["api"]["active"])

    key, url = GET_API("deepseek-chat")
    print("KEY:", key)
    print("URL:", url)