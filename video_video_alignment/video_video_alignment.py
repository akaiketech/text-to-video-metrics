import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from .calculate_fvd import calculate_fvd
from .calculate_ssim import calculate_ssim
import os

class VideoVideoAlignment:
    def __init__(self, generated_video_path, reference_video_path):
        self.generated_video_path = generated_video_path
        self.reference_video_path = reference_video_path

    def load_videos(self, video_path, size=(64, 64), max_frames=30):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size)
        ])
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < max_frames:
            frames.extend([frames[-1]] * (max_frames - len(frames)))
        
        return torch.stack(frames).unsqueeze(0)  # Add batch dimension

    def run(self):
   
        videos1 = self.load_videos(self.generated_video_path)
        videos2 = self.load_videos(self.reference_video_path)
        
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        videos1, videos2 = videos1.to(device), videos2.to(device)
        
   
        fvd_score = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False)
        ssim_score = calculate_ssim(self.generated_video_path, self.reference_video_path)
        
       
        video_name = os.path.basename(self.generated_video_path)
        result_df = pd.DataFrame({
            'video_name': [video_name],
            'reference_video': [os.path.basename(self.reference_video_path)],
            'fvd_score': [fvd_score],
            'ssim_score': [ssim_score],
            'combined_score': [(fvd_score + ssim_score) / 2]  
        })
        
       
        metrics_summary = {
            'metric_name': 'video_alignment',
            'fvd_score': fvd_score,
            'ssim_score': ssim_score,
            'combined_score': (fvd_score + ssim_score) / 2
        }
        
        return result_df, metrics_summary









