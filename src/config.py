from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Database
    database_url: str = Field(
        default="postgresql://marketmind:marketmind_dev@localhost:5432/marketmind"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # LLM
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")

    # App
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Rabat bounding box (south, west, north, east)
    rabat_bbox: tuple[float, float, float, float] = (33.97, -6.87, 34.03, -6.80)
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
