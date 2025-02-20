import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from tqdm import tqdm
from calculate_fvd import calculate_fvd
from calculate_ssim import calculate_ssim


def load_videos(video_paths, size=(64, 64), max_frames=30):
    videos = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (HWC -> CHW, normalized [0,1])
        transforms.Resize(size)  # Resize to the desired size
    ])
    
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = transform(frame)  # Convert to tensor and resize
            frames.append(frame)

            frame_count += 1

        cap.release()

        if len(frames) < max_frames:
            # Pad with last frame if video is shorter than required length
            frames.extend([frames[-1]] * (max_frames - len(frames)))

        videos.append(torch.stack(frames))  # Stack frames into (T, C, H, W)

    videos = torch.stack(videos)  # Convert to tensor (N, T, C, H, W)
    return videos


def compute_video_to_video_quality(generated_video_path, reference_video_path, device, method='styleganv', only_final=False):
   

    videos1 = load_videos(generated_video_path)
    videos2 = load_videos(reference_video_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    videos1, videos2 = videos1.to(device), videos2.to(device)

    result = {}
    only_final = False

    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)

    result['ssim'] = calculate_ssim(generated_video_path, reference_video_path, only_final=only_final)

    print(json.dumps(result, indent=4))

    return result
