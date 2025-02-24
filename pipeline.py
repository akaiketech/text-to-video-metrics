from text_video_alignment.text_to_video_alignment import TextToVideoAlignment
from video_video_alignment.video_video_alignment import VideoVideoAlignment
from video_quality.video_quality import VideoQuality



class Pipeline:
    def __init__(self, config):
        self.text_to_video_alignment = TextToVideoAlignment(config['input_video_directory'], 
                                                          config['output_path'],
                                                          config['captions_list'])
        self.video_video_alignment = VideoVideoAlignment(config['generated_video_path'], 
                                                       config['reference_video_path'])
        self.video_quality = VideoQuality(config['file_path'])

    def create_detailed_report(self, text_video_df, video_quality_df, video_video_df,
                             text_video_metrics, video_quality_metrics, video_video_metrics):
 
        final_df = text_video_df.merge(video_quality_df, on='video_name', how='outer')\
                               .merge(video_video_df, on='video_name', how='outer')
        

        metrics_df = pd.DataFrame([
            text_video_metrics,
            video_quality_metrics,
            video_video_metrics
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
        
      
        report = self.create_detailed_report(
            text_video_df, video_quality_df, video_video_df,
            text_video_metrics, video_quality_metrics, video_video_metrics
        )
        
        print("Pipeline completed")
        print("Detailed results stored in detailed_results.csv")
        print("Metrics summary stored in metrics_summary.csv")
        
        return report



from text_video_alignment.text_to_video_alignment import TextToVideoAlignment
from video_video_alignment.video_video_alignment import VideoVideoAlignment
from video_quality.video_quality import VideoQuality
import pandas as pd
from typing import Dict, List, Optional, Tuple

class Pipeline:
    def __init__(self, config: Dict):
        """
        Initialize pipeline with configurable components.
        
        Args:
            config: Dictionary containing:
                - component_selection: List of components to include ('text_video', 'video_quality', 'video_video')
                - input_video_directory: Path to input videos
                - output_path: Path for output files
                - captions_list: List of captions
                - generated_video_path: Path to generated videos
                - reference_video_path: Path to reference videos
                - file_path: Path for video quality analysis
        """
        self.config = config
        self.components = {}
        self.selected_components = config.get('component_selection', ['text_video', 'video_quality', 'video_video'])
        
        # Initialize only selected components
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

    def create_detailed_report(self, results: Dict[str, Tuple[pd.DataFrame, Dict]]) -> Dict:
        """
        Create a detailed report based on selected components.
        
        Args:
            results: Dictionary containing DataFrames and metrics for each component
            
        Returns:
            Dictionary containing merged results and metrics summary
        """
        # Initialize with first available DataFrame
        final_df = None
        all_metrics = []
        
        # Process each selected component's results
        for component_name, (df, metrics) in results.items():
            if final_df is None:
                final_df = df
            else:
                final_df = final_df.merge(df, on='video_name', how='outer')
            
            metrics['component'] = component_name
            all_metrics.append(metrics)
        
        # Create metrics summary DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save results to files
        if final_df is not None:
            final_df.to_csv("detailed_results.csv", index=False)
        metrics_df.to_csv("metrics_summary.csv", index=False)
        
        # Create report dictionary
        report = {
            'detailed_results': final_df,
            'metrics_summary': metrics_df,
            'components_included': self.selected_components
        }
        
        return report

    def run(self) -> Dict:
        """
        Run the pipeline with selected components.
        
        Returns:
            Dictionary containing detailed results and metrics summary
        """
        results = {}
        
        # Run each selected component
        for component_name, component in self.components.items():
            df, metrics = component.run()
            results[component_name] = (df, metrics)
        
        # Create and return report
        report = self.create_detailed_report(results)
        
        print("Pipeline completed")
        print(f"Components included: {', '.join(self.selected_components)}")
        print("Detailed results stored in detailed_results.csv")
        print("Metrics summary stored in metrics_summary.csv")
        
        return report
        
        