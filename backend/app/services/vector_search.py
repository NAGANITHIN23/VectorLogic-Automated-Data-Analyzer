from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

class VectorSearchService:
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db