import yaml
from app.models import AppConfig
from app.settings import EnvSettings


def load_config() -> tuple[AppConfig, EnvSettings]:
    env = EnvSettings()
    with open(env.config_path, "r") as f:
        raw = yaml.safe_load(f)
    config = AppConfig.model_validate(raw)
    return config, env
