from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from pydantic_models import PipelineConfig, PipelineResponseDataFrame, ComponentResultsResponse, PipelineResultsResponse
import pandas as pd
from pipeline import Pipeline, PipelineResults, ComponentResults

app = FastAPI(
    title="Text-to-Video Alignment Pipeline API",
)

def convert_dataframe_to_response(df: Optional[pd.DataFrame]) -> Optional[PipelineResponseDataFrame]:
    if df is None:
        return None
    
    return PipelineResponseDataFrame(
        columns=df.columns.tolist(),
        data=df.values.tolist()
    )

def convert_component_results(comp_results: Optional[ComponentResults]) -> Optional[ComponentResultsResponse]:
    if comp_results is None:
        return None
    
    return ComponentResultsResponse(
        dataframe=convert_dataframe_to_response(comp_results.dataframe),
        metrics=comp_results.metrics
    )

def convert_pipeline_results(results: PipelineResults) -> PipelineResultsResponse:
    return PipelineResultsResponse(
        text_video_results=convert_component_results(results.text_video_results),
        detailed_results=convert_dataframe_to_response(results.detailed_results)
    )

@app.post("/run-pipeline", response_model=PipelineResultsResponse)
async def run_pipeline(config: PipelineConfig):
    try:
        pipeline = Pipeline(config.dict())
        results = pipeline.run()
        response = convert_pipeline_results(results)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")



if __name__ == "__main__":
    app.run(debug=True, port="6655", host="127.0.0.1")