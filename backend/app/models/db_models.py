from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class JobStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_size_mb = Column(Float)
    s3_key = Column(String)
    
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Integer, default=0)
    
    rows = Column(Integer)
    columns = Column(Integer)
    
    result_s3_key = Column(String)
    analysis_result = Column(Text)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
