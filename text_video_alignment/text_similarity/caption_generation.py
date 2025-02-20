import cv2
import numpy as np
from matplotlib import cm
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import json
import os
import sys


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

def create_frame_list(filepath):
    """
    This function takes in a video filepath.
    Reads in the video frame by frame using cv2.
    Converts the frame into a BLIP suitable format.
    Returns a list of frames
    """
    vidcap = cv2.VideoCapture(filepath)
    success,raw_image = vidcap.read()
    frames = []
    while success:
        frame = vis_processors["eval"](Image.fromarray(raw_image)).unsqueeze(0).to(device)
        frames.append(frame)
        success,raw_image = vidcap.read()
    return frames

def generate_BLIP_captions(frames):
    """
    This function takes in a list of frames in a BLIP suitable format.
    Generates a caption for each frame.
    Returns a dictionary with the key being the frame id and the value the caption for that frame.
    """
    captions = {}
    for i, frame in enumerate(frames):
        captions[i] = model.generate({"image": frame})
    return captions

def run_BLIP_model(filepath):
    """
    This function takes in a video filepath.
    Creates a BLIP suitable frame list using create_frame_list()
    Generates the BLIP captions of the frames using generate_BLIP_captions()
    Returns a dictionary of the captions.
    """
    frames = create_frame_list(filepath)
    captions = generate_BLIP_captions(frames)
    return captions

def is_video(filename, filetypes=[".mp4", ".gif"]):
    """
    Checks if a file is a video based on the file extension.
    Returns True if the file extension matches the selected types else False.
    """
    for filetype in filetypes:
        if filename.endswith(filetype):
            return True, filename.split(filetype)[0]
    return False, None




def generate_captions(input_video_directory, output_path):
    """
    This function takes in a video filepath.
    Generates BLIP captions for the video.
    Returns a dictionary of the captions.
    """
    

    run_model = True
    video_directory = input_video_directory
    output_path = output_path


    complete_caption_dict = {}
    isvideo, real_caption = is_video(video_directory)
    if isvideo:
        complete_caption_dict[real_caption.replace("_", " ")] = run_BLIP_model(video_directory)
    else:
        for model_type in os.listdir(video_directory):
            filepath = f"{video_directory}/{model_type}/"
            complete_caption_dict[model_type] = {}
            print(model_type)
            for video_name in os.listdir(filepath):
                isvideo, real_caption = is_video(video_name)
                if isvideo:
                    complete_caption_dict[model_type][real_caption.replace("_", " ")] = run_BLIP_model(filepath+video_name)

    output_file = open(file=output_path, mode="w", encoding="utf-8")
    json.dump(complete_caption_dict, output_file, indent=4)
    output_file.close()
