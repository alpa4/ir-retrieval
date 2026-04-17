from pydantic_settings import BaseSettings


class EnvSettings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    config_path: str = "/app/config/config.yaml"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
