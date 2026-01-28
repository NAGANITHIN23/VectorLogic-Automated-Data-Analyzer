import redis
from app.config import settings
import json
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        return f"{settings.APP_NAME}:{prefix}:{identifier}"
    
    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        try:
            key = self._generate_key(prefix, identifier)
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    def set(self, prefix: str, identifier: str, value: Any, ttl: int = 3600):
        try:
            key = self._generate_key(prefix, identifier)
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    def delete(self, prefix: str, identifier: str):
        try:
            key = self._generate_key(prefix, identifier)
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
