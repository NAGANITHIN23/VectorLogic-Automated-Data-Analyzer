from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.db_models import AnalysisJob
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{job_id}")
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(AnalysisJob).where(AnalysisJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
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
    
    if job.analysis_result:
        try:
            response["analysis"] = json.loads(job.analysis_result)
        except:
            pass
    
    return response

@router.get("/")
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(AnalysisJob).order_by(desc(AnalysisJob.created_at)).limit(limit).offset(offset)
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    
    jobs_list = []
    for job in jobs:
        job_data = {
            "job_id": job.id,
            "filename": job.filename,
            "status": job.status.value,
            "rows": job.rows,
            "columns": job.columns,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "file_size_mb": job.file_size_mb
        }
        jobs_list.append(job_data)
    
    stmt_count = select(AnalysisJob)
    result_count = await db.execute(stmt_count)
    total_count = len(result_count.scalars().all())
    
    return {
        "jobs": jobs_list,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }
