import os
from celery import Celery
from .pipeline import TextToVideoAlignment
import tempfile
import shutil
import uuid
import os
import json
from loguru import logger



redis_host = os.environ.get("REDIS_HOST", "redis")
redis_port = os.environ.get("REDIS_PORT", "6379")

celery_app = Celery(
    "video_quality_worker",
    broker=f"redis://{redis_host}:{redis_port}/0",
    backend=f"redis://{redis_host}:{redis_port}/0"
)

@celery_app.task(name="alignment_tasks.analyze")
def process_text_video_alignment(video_dir, captions_list,caption_dir):
    
    try:

        logger.info("Captions: %s", captions_list)
        caption_path = os.path.join(caption_dir, f'captions_{uuid.uuid4().hex}.json')
        
        logger.info("Caption path: %s", caption_path)

               
        alignment_analyzer = TextToVideoAlignment(
            input_video_directory=video_dir,
            captions_list=captions_list,
            output_path=caption_path
        )
        
        standardized_df, metrics_summary = alignment_analyzer.run()
        
 
        df_dict = standardized_df.to_dict(orient='records')
        
        return {
            "status": "success",
            "standardized_metrics": df_dict,
            "metrics_summary": metrics_summary
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

    finally:
        logger.info("Cleaning up temporary directories")
        try:
            shutil.rmtree(video_dir)
            shutil.rmtree(caption_dir)
        except Exception:
            pass