from text_similarity.text_similarity_calculation import compute_similarity_score
from text_similarity.caption_generation import generate_captions


class TextToVideoAlignment:
    def __init__(self, input_video_directory,captions_list, output_path):
        self.input_video_directory = input_video_directory
        self.output_path = output_path
        self.captions_list = captions_list
    
    def run(self):
        generate_captions(self.input_video_directory, self.output_path,self.captions_list)
        result=compute_similarity_score(self.output_path)

        return result





