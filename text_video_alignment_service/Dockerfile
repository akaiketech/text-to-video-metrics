# alignment_service/Dockerfile
FROM python:3.9
WORKDIR /app/text_video_alignment_service
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "-Q", "alignment_queue"]