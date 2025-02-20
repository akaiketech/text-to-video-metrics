from text_similarity.text_similarity_calculation import compute_similarity_score
from text_similarity.caption_generation import generate_captions



def compute_text_to_video_alignment(input_video_directory, output_path):
    caption_path=generate_captions(input_video_directory, output_path)
    result=compute_similarity_score(caption_path)

    return result


