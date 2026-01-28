from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.db_models import AnalysisJob, JobStatus
from app.tasks.analysis import analyze_csv_task
import uuid
import logging
import pandas as pd
import io

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def upload_and_analyze(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > 50:
        raise HTTPException(status_code=400, detail="File too large. Maximum 50MB")
    
    job_id = str(uuid.uuid4())
    
    job = AnalysisJob(
        id=job_id,
        filename=file.filename,
        file_size_mb=round(file_size_mb, 2),
        s3_key=f"local/{job_id}/{file.filename}",
        status=JobStatus.PENDING,
        progress=0
    )
    
    db.add(job)
    await db.commit()
    
    csv_data = contents.decode('utf-8')
    analyze_csv_task.delay(job_id, csv_data, file.filename)
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "pending",
        "message": "Analysis started. Use job_id to check status."
    }

@router.post("/analyze-direct/")
async def analyze_direct(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    from app.services.llm_orchestrator import LLMOrchestrator
    
    contents = await file.read()
    csv_data = contents.decode('utf-8')
    csv_buffer = io.StringIO(csv_data)
    df = pd.read_csv(csv_buffer)
    
    orchestrator = LLMOrchestrator(db=None)
    result = await orchestrator.analyze(df, file.filename)
    
    return {
        "success": result['success'],
        "profile_data": result['profile_data'],
        "analysis_text": result['analysis_text']
    }