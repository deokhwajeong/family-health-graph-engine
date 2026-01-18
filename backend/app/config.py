"""
Configuration management for the Household Health Graph API.

Supports environment-based configuration with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    app_name: str = "Household Health Graph API"
    app_version: str = "0.2.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # ML Settings
    embedding_dim: int = 16
    num_layers: int = 2
    anomaly_contamination: float = 0.1
    anomaly_sensitivity: float = 0.5
    
    # Data Settings
    default_days: int = 60
    max_days: int = 365
    min_days: int = 1
    
    # Model Settings
    model_save_interval: int = 100  # Save model every N predictions
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    
    class Config:
        """Pydantic config for environment variable loading."""
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields


# Create global settings instance
settings = Settings()
