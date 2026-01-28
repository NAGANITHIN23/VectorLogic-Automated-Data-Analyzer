from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    APP_NAME: str = "Automated Data Analyzer"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET: str
    
    MAX_FILE_SIZE_MB: int = 100
    UPLOAD_ALLOWED_EXTENSIONS: List[str] = ["csv"]
    
    CORS_ORIGINS: List[str] = ["*"]
    
    VECTOR_SEARCH_ENABLED: bool = True
    VECTOR_SIMILARITY_THRESHOLD: float = 0.75
    MAX_SIMILAR_RESULTS: int = 5
    
    ENABLE_METRICS: bool = True
    SENTRY_DSN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
