import pandas as pd
from pydantic import BaseModel
from typing import Optional, Dict
from text_video_alignment.text_to_video_alignment import TextToVideoAlignment
from video_video_alignment.video_video_alignment import VideoVideoAlignment
from video_quality.video_quality import VideoQuality


class PipelineResults(BaseModel):
    """Pydantic model to structure pipeline results."""
    text_video_df: Optional[pd.DataFrame] = None
    text_video_metrics: Optional[Dict] = None
    video_quality_df: Optional[pd.DataFrame] = None
    video_quality_metrics: Optional[Dict] = None
    video_video_df: Optional[pd.DataFrame] = None
    video_video_metrics: Optional[Dict] = None


class Pipeline:
    def __init__(self, config):
        self.text_to_video_alignment = TextToVideoAlignment(config['input_video_directory'], 
                                                          config['output_path'],
                                                          config['captions_list'])
        self.video_video_alignment = VideoVideoAlignment(config['generated_video_path'], 
                                                       config['reference_video_path'])
        self.video_quality = VideoQuality(config['file_path'])

    def generate_report(self, results: PipelineResults):
 
        final_df = results.text_video_df.merge(results.video_quality_df, on='video_name', how='outer')\
                               .merge(results.video_video_df, on='video_name', how='outer')
        

        metrics_df = pd.DataFrame([
            results.text_video_metrics,
            results.video_quality_metrics,
            results.video_video_metrics
        ])
        
        final_df.to_csv("detailed_results.csv", index=False)
        metrics_df.to_csv("metrics_summary.csv", index=False)
        
        report = {
            'detailed_results': final_df,
            'metrics_summary': metrics_df
        }
        
        return report

    def run(self):
     
        text_video_df, text_video_metrics = self.text_to_video_alignment.run()
        video_quality_df, video_quality_metrics = self.video_quality.run()
        video_video_df, video_video_metrics = self.video_video_alignment.run()
        
        results = PipelineResults(
                        text_video_df=text_video_df,
                        text_video_metrics=text_video_metrics,
                        video_quality_df=video_quality_df, 
                        video_quality_metrics=video_quality_metrics,
                        video_video_df=video_video_df, 
                        video_video_metrics=video_video_metrics
                )
        report = self.generate_report(results)
        
        print("Pipeline completed")
        print("Detailed results stored in detailed_results.csv")
        print("Metrics summary stored in metrics_summary.csv")
        
        return report
        
        
