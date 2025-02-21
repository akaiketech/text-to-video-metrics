from text_video_alignment.text_to_video_alignment import TextToVideoAlignment
from video_video_alignment.video_video_alignment import VideoVideoAlignment
from video_quality.video_quality import VideoQuality



class Pipeline:
    def __init__(self, config):
        self.text_to_video_alignment = TextToVideoAlignment(config['input_video_directory'], config['output_path'],config['captions_list'])
        self.video_video_alignment = VideoVideoAlignment(config['generated_video_path'], config['reference_video_path'])
        self.video_quality = VideoQuality(config['file_path'])


    def run(self):
        text_to_video_result=self.text_to_video_alignment.run()
        video_quality_result=self.video_quality.run()
        video_video_result=self.video_video_alignment.run()

        result_df=pd.DataFrame({
            'text_to_video':text_to_video_result,
            'video_quality':video_quality_result,
            'video_video':video_video_result
        })

        return result_df
        
        