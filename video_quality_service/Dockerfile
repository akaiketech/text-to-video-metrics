# video_quality_service/Dockerfile
FROM python:3.9
WORKDIR /app/video_quality_service
COPY requirements.txt .
RUN pip install -r requirements.txt
# Install specific torch version
RUN pip install torch==1.9.0 torchvision==0.10.0
COPY . .
CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "-Q", "video_quality_queue"]