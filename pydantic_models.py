from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd

class ComponentResults(BaseModel):
    dataframe: Optional[pd.DataFrame] = Field(default=None, description="Component output DataFrame")
    metrics: Optional[Dict] = Field(default=None, description="Component metrics dictionary")
    
    class Config:
        arbitrary_types_allowed = True

class PipelineResults(BaseModel):
    text_video_results: Optional[ComponentResults] = Field(
        default=None,
        description="Results from text-to-video alignment component"
    )
    detailed_results: Optional[pd.DataFrame] = Field(
        default=None,
        description="DataFrame containing component results"
    )
    
    class Config:
        arbitrary_types_allowed = True

class PipelineConfig(BaseModel):
    input_video_directory: str = Field(..., description="Directory containing input videos")
    output_path: str = Field(..., description="Path to save output files")
    captions_list: List[str] = Field(..., description="List of caption files")

class PipelineResponseDataFrame(BaseModel):
    columns: List[str]
    data: List[List]

class ComponentResultsResponse(BaseModel):
    dataframe: Optional[PipelineResponseDataFrame] = None
    metrics: Optional[Dict] = None

class PipelineResultsResponse(BaseModel):
    text_video_results: Optional[ComponentResultsResponse] = None
    detailed_results: Optional[PipelineResponseDataFrame] = None