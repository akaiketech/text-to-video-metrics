from fastapi import *
from typing import List
from schemas import TextVideoAlignmentRequest
from text_video_alignment_service.pipeline import TextToVideoAlignment
from text_video_alignment_service.worker import process_text_video_alignment, celery_app
import tempfile
import shutil
import uuid
import os
import json
from loguru import logger

router = APIRouter(
    tags=["Text-Video Alignment"]
)

@router.post("/evaluate")
async def evaluate_text_video_alignment(
    files: List[UploadFile] = File(...),
    captions: str = Form(...),
):
    logger.info("Evaluating text-video alignment")
    try:
        captions_list = json.loads(captions)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Captions must be a valid JSON list")
    
    
    if not files:
        raise HTTPException(status_code=400, detail="No video files uploaded")
    
    if not captions or not isinstance(captions_list, list):
        raise HTTPException(status_code=400, detail="Captions list is required")
    

    temp_video_dir = tempfile.mkdtemp(prefix='video_input_')
    temp_caption_dir = tempfile.mkdtemp(prefix='captions_')
    
    try:
     
        video_paths = []
        for file in files:
            file_path = os.path.join(temp_video_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            video_paths.append(file_path)
        
        logger.info("Captions: %s", captions_list)
        caption_path = os.path.join(temp_caption_dir, f'captions_{uuid.uuid4().hex}.json')
        
        logger.info("Caption path: %s", caption_path)
     
        alignment_pipeline = TextToVideoAlignment(
            input_video_directory=temp_video_dir,
            captions_list=captions_list,
            output_path=caption_path
        )
        logger.info("Alignment pipeline started")
    
        standardized_df, metrics_summary = alignment_pipeline.run()

        logger.info("Alignment pipeline completed")


        return {
            "metrics": metrics_summary,
            "detailed_results": standardized_df.to_dict(orient='records')
        }

        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        logger.info("Cleaning up temporary directories")
        try:
            shutil.rmtree(temp_video_dir)
            shutil.rmtree(temp_caption_dir)
        except Exception:
            pass




@router.get("/evaluate/celery")
async def evaluate_text_video_alignment(
    files: List[UploadFile] = File(...),
    captions: str = Form(...),
):
    logger.info("Evaluating text-video alignment")
    try:
        captions_list = json.loads(captions)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Captions must be a valid JSON list")
    
    
    if not files:
        raise HTTPException(status_code=400, detail="No video files uploaded")
    
    if not captions or not isinstance(captions_list, list):
        raise HTTPException(status_code=400, detail="Captions list is required")
    

    temp_video_dir = tempfile.mkdtemp(prefix='video_input_')
    temp_caption_dir = tempfile.mkdtemp(prefix='captions_')
    
    try:
     
        video_paths = []
        for file in files:
            file_path = os.path.join(temp_video_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            video_paths.append(file_path)
        
        task = process_text_video_alignment.delay(temp_video_dir, captions_list, temp_caption_dir)
        
        logger.info(f"Submitted task with ID: {task.id}")
        
        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "task_id": task.id,
                "status": "Processing",
                "message": "Text-video alignment evaluation is being processed"
            }
        )
    
    except Exception as e:
    
        try:
            shutil.rmtree(temp_video_dir)
            shutil.rmtree(temp_caption_dir)
        except Exception:
            pass
        
        logger.error(f"Error in evaluate_text_video_alignment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            "task_id": task_id,
            "status": "PENDING",
            "message": "Task is pending execution"
        }
    elif task.state == 'FAILURE':
        response = {
            "task_id": task_id,
            "status": "FAILURE",
            "message": str(task.info)
        }
    elif task.state == 'SUCCESS':
        response = {
            "task_id": task_id,
            "status": "SUCCESS",
            "results": task.result
        }
    else:
        response = {
            "task_id": task_id,
            "status": task.state,
            "message": "Task is in progress"
        }
    
    return response