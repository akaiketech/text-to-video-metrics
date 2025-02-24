from .video_naturalness_prediction.video_processing import *
import pandas as pd




class VideoQuality:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def run(self):
      
        result_df = calculate_naturalness_score(self.file_path)
     
        confidence_score = result_df['confidence_score'].iloc[0]
        video_name = result_df['video_name'].iloc[0]
 
        standardized_df = pd.DataFrame({
            'video_name': video_name,
            'quality_score': confidence_score,
            'naturalness_level': ['high' if confidence_score >= 0.7 
                                else 'medium' if confidence_score >= 0.4 
                                else 'low']
        })
        
        metrics_summary = {
            'metric_name': 'video_quality',
            'quality_score': confidence_score,
            'naturalness_assessment': 'high' if confidence_score >= 0.7 
                                    else 'medium' if confidence_score >= 0.4 
                                    else 'low'
        }
        
        return standardized_df, metrics_summary

