# alignment_service/Dockerfile
FROM python:3.9
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm
RUN pip install opencv-python
RUN pip install numpy
RUN pip install loguru
RUN apt update && apt install -y libgl1-mesa-glx
COPY . .
CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "-Q", "alignment_queue"]