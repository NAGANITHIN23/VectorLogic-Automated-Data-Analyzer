from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class JobStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatusEnum
    progress: int
    created_at: datetime
    completed_at: Optional[datetime]
    result_url: Optional[str]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class AnalysisResultResponse(BaseModel):
    analysis_text: str
    profile_data: Dict[str, Any]
    visualizations: List[str]
    recommendations: List[Dict[str, str]]
    similar_datasets_found: int
