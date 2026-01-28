from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.db_models import AnalysisJob
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{job_id}")
async def get_analysis_result(job_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(AnalysisJob).where(AnalysisJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "rows": job.rows,
        "columns": job.columns,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message
    }
