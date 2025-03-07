from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class TextVideoAlignmentRequest(BaseModel):
    captions: List[str] = Field(..., 
        description="List of captions or text prompts to compare with videos",
        min_length=1,
        example=["A cat playing", "A dog running in a park"]
    )

class AlignmentMetrics(BaseModel):
    metric_name: str
    overall_similarity_score: float
    best_frame_score: float
    alignment_quality: str

class TextVideoAlignmentResponse(BaseModel):
    metrics: AlignmentMetrics
    detailed_results: dict


class NaturalnessLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class VideoQualityMetrics(BaseModel):
    """Metrics for video quality assessment"""
    metric_name: str
    quality_score: float
    naturalness_assessment: NaturalnessLevel

class VideoQualityResult(BaseModel):
    """Detailed result for a single video"""
    video_name: str
    quality_score: float
    naturalness_level: NaturalnessLevel

class VideoQualityResponse(BaseModel):
    """Response model for video quality assessment"""
    metrics: VideoQualityMetrics
    detailed_results: VideoQualityResult

    class Config:
        arbitrary_types_allowed = True 