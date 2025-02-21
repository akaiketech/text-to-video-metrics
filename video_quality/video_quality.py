from Video_Naturalness.video_processing import *
import json



class VideoQuality:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def run(self):
        result_df=calculate_naturalness_score(self.file_path)
        return result_df

