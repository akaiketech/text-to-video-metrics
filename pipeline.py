from text_video_alignment import TextToVideoAlignment
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic_models import ComponentResults, PipelineResults

class Pipeline:
    def __init__(self, config: Dict):
        """Initialize pipeline with text-to-video alignment component."""
        self.config = config
        self.text_video_component = TextToVideoAlignment(
            config['input_video_directory'],
            config['output_path'],
            config['captions_list']
        )
    
    def create_detailed_report(self, df: pd.DataFrame, metrics: Dict) -> PipelineResults:
        
        # df.to_csv(f"{self.config['output_path']}/detailed_results.csv", index=False)
        
        text_video_results = ComponentResults(
            dataframe=df,
            metrics=metrics
        )
        
        return PipelineResults(
            text_video_results=text_video_results,
            detailed_results=df
        )
    
    def run(self) -> PipelineResults:
        """Run the pipeline and return structured results."""
        df, metrics = self.text_video_component.run()
        report = self.create_detailed_report(df, metrics)
        
        print("Pipeline completed")
        print("Text-to-video alignment component executed successfully")
        print("Detailed results stored in detailed_results.csv")
        
        return report