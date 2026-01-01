from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
if not ENV_FILE.exists():
    ENV_FILE = BASE_DIR / ".env.example"
    if not ENV_FILE.exists():
        raise OSError(f"Neither `.env` nor `.env.example` were found in {BASE_DIR}")


class PydanticSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    TRAINING_DIR: Path
    VALIDATION_DIR: Path
    RANDOM_SEED: int = 1337


p_env = PydanticSettings()  # type:ignore[reportCallIssue]
