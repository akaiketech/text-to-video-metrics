import cv2
import numpy as np
from PIL import Image
from t2v_metrics import VQAScore, CLIPScore

class VideoScorer:
    def __init__(self, model_type='vqascore', frame_sample=16):
        self.frame_sample = frame_sample
        self.model_type = model_type
        self.scorer = self._load_scorer()

    def _load_scorer(self):
        if self.model_type == 'vqascore':
            return VQAScore(model='clip-flant5-xl')
        elif self.model_type == 'clipscore':
            return CLIPScore(model='openai:ViT-L-14')
        else:
            raise ValueError("Invalid model type, choose 'vqascore' or 'clipscore'")

    def extract_frames(self, video_path):
        """Extract uniformly sampled frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, self.frame_sample, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def score_video(self, video_path, text, aggregation='mean'):
        """Score video-text alignment using frame aggregation"""

        frames = self.extract_frames(video_path)


        text_list = [text] * len(frames)
        scores = self.scorer(images=frames, texts=text_list)


        if aggregation == 'mean':
            return np.mean(scores)
        elif aggregation == 'min':
            return np.min(scores)
        elif aggregation == 'max':
            return np.max(scores)
        else:
            raise ValueError("Invalid aggregation method")


scorer = VideoScorer(model_type='vqascore', frame_sample=16)
video_score = scorer.score_video(
    video_path="/content/0214.mp4",
    text="Girl Talking infront of camera",
    aggregation='mean'
)
print(f"Video-Text Alignment Score: {video_score:.4f}")