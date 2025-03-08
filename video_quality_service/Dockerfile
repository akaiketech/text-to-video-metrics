# video_quality_service/Dockerfile
FROM python:3.9
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install scikit-video
RUN apt update && apt install -y libgl1-mesa-glx
RUN pip install keras tensorflow
RUN pip install "numpy<2.0"
RUN pip install tf-keras
COPY . .

CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "-Q", "video_quality_queue"]