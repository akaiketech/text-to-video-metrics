import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from tqdm import tqdm
from calculate_fvd import calculate_fvd
from calculate_ssim import calculate_ssim





class VideoVideoAlignment:
    def __init__(self, generated_video_path, reference_video_path):
        self.generated_video_path = generated_video_path
        self.reference_video_path = reference_video_path

    def load_videos(self,video_paths, size=(64, 64), max_frames=30):
        videos = []
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size)
        ])
        
        for path in video_paths:
            cap = cv2.VideoCapture(path)
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

            videos.append(torch.stack(frames)) 

        videos = torch.stack(videos)
        return videos

    def run(self, generated_video_path, reference_video_path, device, method='styleganv', only_final=False):
   

        videos1 = self.load_videos(generated_video_path)
        videos2 = self.load_videos(reference_video_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        videos1, videos2 = videos1.to(device), videos2.to(device)

        result = {}
        only_final = False

        result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)

        result['ssim'] = calculate_ssim(generated_video_path, reference_video_path, only_final=only_final)

        print(json.dumps(result, indent=4))

        return result









