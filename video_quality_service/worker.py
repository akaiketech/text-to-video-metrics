import os
from celery import Celery
from .pipeline import VideoQuality



redis_host = os.environ.get("REDIS_HOST", "redis")
redis_port = os.environ.get("REDIS_PORT", "6379")

celery_app = Celery(
    "video_quality_worker",
    broker=f"redis://{redis_host}:{redis_port}/0",
    backend=f"redis://{redis_host}:{redis_port}/0"
)


@celery_app(name="video_quality_assessment_task")
def process_video_quality(file_path):
    """
    Celery task to process video quality assessment asynchronously
    
    Args:
        file_path (str): Path to the video file to assess
        
    Returns:
        dict: Results containing metrics and detailed results
    """
    logger.info(f"Starting video quality assessment task for: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at path: {file_path}")
            
       
        video_quality_assessment = VideoQuality(file_path)
        standardized_df, metrics_summary = video_quality_assessment.run()
        
   
        result = standardized_df.to_dict(orient='records')[0]
        
        logger.info("Video quality assessment completed successfully")
        
        return {
            "metrics": {
                "metric_name": "video_quality",
                "quality_score": metrics_summary['quality_score'],
                "naturalness_assessment": metrics_summary['naturalness_assessment']
            },
            "detailed_results": result
        }
        
    except Exception as e:
        logger.error(f"Error in video quality assessment task: {str(e)}")
        raise
        
    finally:
        if file_path:
            file_path = file_path.decode() if isinstance(file_path, bytes) else file_path
            
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Removed temporary file after failed task: {file_path}")
        

