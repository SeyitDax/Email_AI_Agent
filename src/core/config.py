"""
Configuration Management - BOILERPLATE
Centralized configuration for the Email AI Agent
"""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application Settings
    app_name: str = "Email AI Agent"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=500, env="OPENAI_MAX_TOKENS")
    
    # Database Configuration
    database_url: str = Field(default="sqlite+aiosqlite:///email_agent.db", env="DATABASE_URL")
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    db_echo: bool = Field(default=False, env="DB_ECHO")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Email Service Configuration
    use_gmail_api: bool = Field(default=False, env="USE_GMAIL_API")
    use_smtp: bool = Field(default=False, env="USE_SMTP")
    gmail_credentials_path: Optional[str] = Field(default=None, env="GMAIL_CREDENTIALS_PATH")
    gmail_token_path: Optional[str] = Field(default=None, env="GMAIL_TOKEN_PATH")
    email_check_interval: int = Field(default=60, env="EMAIL_CHECK_INTERVAL")  # seconds
    
    # SMTP Configuration
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    
    # Email Defaults
    default_from_name: str = Field(default="Email AI Agent", env="DEFAULT_FROM_NAME")
    default_from_email: str = Field(default="noreply@example.com", env="DEFAULT_FROM_EMAIL")
    escalation_email: str = Field(default="admin@example.com", env="ESCALATION_EMAIL")
    
    # Queue Configuration
    task_queue_name: str = Field(default="email_processing", env="TASK_QUEUE_NAME")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: int = Field(default=5, env="RETRY_DELAY")  # seconds
    
    # Classification Thresholds
    confidence_threshold_send: float = Field(default=0.85, env="CONFIDENCE_THRESHOLD_SEND")
    confidence_threshold_review: float = Field(default=0.65, env="CONFIDENCE_THRESHOLD_REVIEW")
    escalation_threshold: float = Field(default=0.5, env="ESCALATION_THRESHOLD")
    
    # Performance Settings
    max_emails_per_batch: int = Field(default=50, env="MAX_EMAILS_PER_BATCH")
    processing_timeout: int = Field(default=30, env="PROCESSING_TIMEOUT")  # seconds
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Security Settings
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Monitoring & Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()