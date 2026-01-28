from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from app.config import settings
from app.models.db_models import Base
import logging

logger = logging.getLogger(__name__)

_async_engine = None
_async_session_maker = None

def get_async_engine():
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
    return _async_engine

def get_async_session_maker():
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False
        )
    return _async_session_maker

sync_database_url = settings.DATABASE_URL.replace('+asyncpg', '')
sync_engine = create_engine(
    sync_database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False
)

async def get_db():
    AsyncSessionLocal = get_async_session_maker()
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

def get_sync_session():
    return SyncSessionLocal()

async def init_db():
    try:
        engine = get_async_engine()
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
