from Video_Naturalness.video_processing import *




def compute_video_quality(file_path):

    result_df=calculate_naturalness_score(file_path)
    return result_df