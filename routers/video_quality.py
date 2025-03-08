import os
import uuid
import shutil
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, Field, validator
from video_quality_service.pipeline import VideoQuality
from video_quality_service.worker import process_video_quality, celery_app
from loguru import logger
from typing import Optional

router = APIRouter(
    tags=["Video Quality Assessment"]
)

@router.post("/evaluate")
async def evaluate_video_quality(
    file: UploadFile = File(...)
):
    """
    Assess quality of uploaded video file
    
    - Upload single video file
    - Returns quality metrics for the video
    """

    if not file:
        raise HTTPException(status_code=400, detail="No video file uploaded")
    
    allowed_types = [
        'video/mp4', 
        'video/mpeg', 
        'video/quicktime', 
        'video/x-msvideo'
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported video file type")
    
  
    temp_video_dir = tempfile.mkdtemp(prefix='video_quality_')
    
    try:
        file_path = os.path.join(temp_video_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
 
        # Use the VideoQuality pipeline for assessment
        video_quality_assessment = VideoQuality(file_path)
        standardized_df, metrics_summary = video_quality_assessment.run()
        result = standardized_df.to_dict(orient='records')[0]
        
        return {
            "metrics": {
                "metric_name": "video_quality",
                "quality_score": metrics_summary['quality_score'],
                "naturalness_assessment": metrics_summary['naturalness_assessment']
            },
            "detailed_results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
 
        try:
            shutil.rmtree(temp_video_dir)
        except Exception:
            pass

@router.post("/evaluate/celery")
async def evaluate_video_quality_async(
    file: UploadFile = File(...)
):
    """
    Asynchronous endpoint to assess quality of uploaded video file
    
    - Upload single video file
    - Returns a task ID for tracking progress
    - Check task status using /task/{task_id} endpoint
    """
    
    if not file:
        raise HTTPException(status_code=400, detail="No video file uploaded")
    
    allowed_types = [
        'video/mp4', 
        'video/mpeg', 
        'video/quicktime', 
        'video/x-msvideo'
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported video file type")
    
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            # Copy uploaded file to the temporary file
            shutil.copyfileobj(file.file, temp_file)
            video_path = temp_file.name
        
        logger.info(f"Saved video to temporary file: {video_path}")
        
        task = process_video_quality.delay(video_path)
        logger.info(f"Submitted video quality assessment task: {task.id}")
        
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task.id,
                "status": "Processing",
                "message": "Video quality assessment is being processed"
            }
        )
    except Exception as e:
     
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except Exception:
                pass
        
        logger.error(f"Error in evaluate_video_quality_async: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_video_quality_result(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Task is still processing"
        }
    elif task_result.state == 'FAILURE':
        return {
            "task_id": task_id,
            "status": "error",
            "message": str(task_result.info)
        }
    elif task_result.state == 'SUCCESS':
        result = task_result.get()
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    return {
        "task_id": task_id,
        "status": task_result.state,
        "message": "Task status unknown"
    }