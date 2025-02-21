from text_video_alignment.text_similarity.text_similarity_calculation import compute_similarity_score
from text_video_alignment.text_similarity.caption_generation import generate_captions


class TextToVideoAlignment:
    def __init__(self, input_video_directory,captions_list, output_path):
        self.input_video_directory = input_video_directory
        self.output_path = output_path
        self.captions_list = captions_list
    
    def run(self):
        generate_captions(self.input_video_directory, self.output_path,self.captions_list)
        result_df=compute_similarity_score(self.output_path)

        avg_score = result_df['overall_avg_score'].iloc[0]
        best_score = result_df['best_score'].max()


        standardized_df = result_df
        metrics_summary = {
            'metric_name': 'text_video_alignment',
            'overall_similarity_score': avg_score,
            'best_frame_score': best_score,
            'alignment_quality': 'high' if avg_score >= 0.7 
                               else 'medium' if avg_score >= 0.4 
                               else 'low'
        }
        
        return standardized_df, metrics_summary





