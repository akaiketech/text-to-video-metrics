# JioStar Text-to-Video Metrics  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aL-3bHvYGRtBBsK1hDvg-AzBDvGAR1Nf?usp=sharing)

A  evaluation pipeline for text-to-video generation systems that assesses alignment, quality, and realism metrics.

## Overview

This pipeline evaluates generated videos using three core assessment dimensions:
1. **Text-to-Video Alignment**: How well video content matches input text prompts
2. **Video Quality Assessment**: Naturalness and realism of generated video frames
3. **Video-to-Video Alignment**: Similarity between generated videos and reference videos

## Metrics Summary

### Text-to-Video Alignment
Uses BLIP model to generate captions from video frames and compares them with original text prompts through semantic similarity scores.

### Video Quality Assessment
Extracts statistical features (texture, sharpness, color distribution, spectral properties, entropy, contrast) and uses an XGBoost or AdaBoost classifier to predict naturalness.

### Video-to-Video Alignment
Implements Fréchet Video Distance (FVD) and Structural Similarity Index (SSIM) to measure the distribution similarity and structural alignment between generated and reference videos.

## Installation

```bash
git clone https://github.com/your-username/akaiketech-text-to-video-metrics.git
cd akaiketech-text-to-video-metrics
pip install -r requirements.txt
```

## Usage

1. Configure your evaluation parameters:

```python
config = {
    # Text-to-Video Alignment parameters
    'input_video_directory': 'path/to/generated_videos/',
    'output_path': 'path/to/save/results/',
    'captions_list': ['list', 'of', 'text', 'prompts'],
    
    # Video-Video Alignment parameters
    'generated_video_path': 'path/to/generated_videos/',
    'reference_video_path': 'path/to/reference_videos/',
    
    # Video Quality parameters
    'file_path': 'path/to/videos/for/quality/assessment/'
}
```

2. Run the pipeline:

```python
from pipeline import Pipeline

pipeline = Pipeline(config)
results = pipeline.run()
```

## Output

The pipeline generates two CSV files:
- `detailed_results.csv`: Per-video metrics for all evaluation dimensions
- `metrics_summary.csv`: Aggregated metrics across all videos

## Component Details

### Text-to-Video Alignment
Evaluates how well the generated video content aligns with the original text prompt using:
- Frame-by-frame caption generation with BLIP
- Semantic similarity calculation between generated captions and original text prompt
- Temporal alignment analysis through frame sequence evaluation

### Video Quality Assessment
Evaluates the naturalness and realism of generated videos through:
- Statistical feature extraction (texture, sharpness, color distribution)
- Frequency domain analysis (spectral properties)
- Information theory metrics (entropy)
- Visual perception metrics (contrast)
- Feature-based analysis (ORB features, blob detection)
- XGBoost classifier trained on natural vs. generated content

### Video-to-Video Alignment
Compares generated videos with reference videos using:
- Fréchet Video Distance (FVD): Measures the distribution similarity in feature space
- Structural Similarity Index (SSIM): Measures frame-by-frame structural similarity

## Directory Structure

```
akaiketech-text-to-video-metrics/
├── pipeline.py                     # Main pipeline implementation
├── requirements.txt                # Dependencies
├── text_video_alignment/           # Text-to-video alignment components
├── video_quality/                  # Video quality assessment components
└── video_video_alignment/          # Video-to-video comparison components
```



