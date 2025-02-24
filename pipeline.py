from text_video_alignment.text_to_video_alignment import TextToVideoAlignment
from video_video_alignment.video_video_alignment import VideoVideoAlignment
from video_quality.video_quality import VideoQuality
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
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
    video_quality_results: Optional[ComponentResults] = Field(
        default=None,
        description="Results from video quality analysis component"
    )
    video_video_results: Optional[ComponentResults] = Field(
        default=None,
        description="Results from video-to-video alignment component"
    )
    components_included: List[str] = Field(
        default_factory=list,
        description="List of components included in the pipeline run"
    )
    detailed_results: Optional[pd.DataFrame] = Field(
        default=None,
        description="Merged DataFrame containing all component results"
    )
    metrics_summary: Optional[pd.DataFrame] = Field(
        default=None,
        description="Summary DataFrame of all component metrics"
    )

    class Config:
        arbitrary_types_allowed = True 


class Pipeline:
    def __init__(self, config: Dict):
        """Initialize pipeline with configurable components."""
        self.config = config
        self.components = {}
        self.selected_components = config.get('component_selection', ['text_video', 'video_quality', 'video_video'])
     
        if 'text_video' in self.selected_components:
            self.components['text_video'] = TextToVideoAlignment(
                config['input_video_directory'],
                config['output_path'],
                config['captions_list']
            )
            
        if 'video_quality' in self.selected_components:
            self.components['video_quality'] = VideoQuality(
                config['file_path']
            )
            
        if 'video_video' in self.selected_components:
            self.components['video_video'] = VideoVideoAlignment(
                config['generated_video_path'],
                config['reference_video_path']
            )

    def create_detailed_report(self, results: Dict[str, Tuple[pd.DataFrame, Dict]]) -> PipelineResults:
        """Create a detailed report using Pydantic models."""
        final_df = None
        all_metrics = []
        component_results = {}

      
        for component_name, (df, metrics) in results.items():
         
            component_results[f"{component_name}_results"] = ComponentResults(
                dataframe=df,
                metrics=metrics
            )
            
            # Merge DataFrames
            if final_df is None:
                final_df = df
            else:
                final_df = final_df.merge(df, on='video_name', how='outer')
            
            metrics['component'] = component_name
            all_metrics.append(metrics)

        metrics_df = pd.DataFrame(all_metrics)
        
        # Save results to files
        if final_df is not None:
            final_df.to_csv("detailed_results.csv", index=False)
        metrics_df.to_csv("metrics_summary.csv", index=False)
        
     
        return PipelineResults(
            text_video_results=component_results.get('text_video_results'),
            video_quality_results=component_results.get('video_quality_results'),
            video_video_results=component_results.get('video_video_results'),
            components_included=self.selected_components,
            detailed_results=final_df,
            metrics_summary=metrics_df
        )

    def run(self) -> PipelineResults:
        """Run the pipeline and return structured results."""
        results = {}
        
        for component_name, component in self.components.items():
            df, metrics = component.run()
            results[component_name] = (df, metrics)
        
        report = self.create_detailed_report(results)
        
        print("Pipeline completed")
        print(f"Components included: {', '.join(self.selected_components)}")
        print("Detailed results stored in detailed_results.csv")
        print("Metrics summary stored in metrics_summary.csv")
        
        return report
        
        