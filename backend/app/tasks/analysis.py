from celery import Task
from app.tasks.worker import celery
from app.database import get_sync_session
from app.models.db_models import AnalysisJob, JobStatus
import pandas as pd
import logging
import json
import io
from datetime import datetime

logger = logging.getLogger(__name__)

@celery.task(name='app.tasks.analysis.analyze_csv_task', bind=True)
def analyze_csv_task(self, job_id: str, csv_data: str, filename: str) -> dict:
    import asyncio
    from app.services.llm_orchestrator import LLMOrchestrator
    
    session = get_sync_session()
    
    try:
        job = session.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        job.status = JobStatus.PROCESSING
        job.progress = 20
        session.commit()
        
        csv_buffer = io.StringIO(csv_data)
        df = pd.read_csv(csv_buffer)
        
        job.progress = 30
        session.commit()
        
        logger.info(f"Running NEW analysis with LLMOrchestrator for job {job_id}")
        
        orchestrator = LLMOrchestrator(db=None)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis_result = loop.run_until_complete(orchestrator.analyze(df, filename))
        loop.close()
        
        job.progress = 90
        session.commit()
        
        if not analysis_result.get('success', False):
            raise Exception(analysis_result.get('error', 'Analysis failed'))
        
        profile_data = analysis_result['profile_data']
        
        result_dict = {
            'llm_insights': analysis_result['analysis_text'],
            'basic_info': profile_data.get('basic_info', {}),
            'missing_analysis': profile_data.get('missing_analysis', {}),
            'missing_patterns': profile_data.get('missing_patterns', {}),
            'missing_mechanisms': profile_data.get('missing_mechanisms', {}),
            'normality_tests': profile_data.get('normality_tests', {}),
            'intelligent_types': profile_data.get('intelligent_types', {}),
            'numeric_stats': profile_data.get('numeric_stats', {}),
            'outliers': profile_data.get('outliers', {}),
            'outliers_advanced': profile_data.get('outliers_advanced', {}),
            'data_quality': profile_data.get('data_quality', {}),
            'preprocessing': profile_data.get('preprocessing', {}),
            'feature_engineering': profile_data.get('feature_engineering', {}),
            'correlations': profile_data.get('correlations', {}),
            'categorical_analysis': profile_data.get('categorical_analysis', {}),
            'recommendations': profile_data.get('recommendations', [])
        }
        
        logger.info(f"NEW Analysis complete. Keys in result: {list(result_dict.keys())}")
        
        job.analysis_result = json.dumps(result_dict)
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.rows = len(df)
        job.columns = len(df.columns)
        session.commit()
        
        logger.info(f"NEW Analysis saved successfully for job {job_id}")
        
        return {
            'job_id': job_id,
            'status': 'completed',
            'analysis': result_dict
        }
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {str(e)}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        session.commit()
        raise
    
    finally:
        session.close()