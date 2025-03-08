# Python example
import requests

url = "http://127.0.0.1:8000/text-video-alignment/evaluate"
files = [
    ('files', ('video1.mp4', open(r'C:\Users\Akaike\Akaike-repos\jio-t2v\text2metrics\output_7.mp4', 'rb'), 'video/mp4')),
]


form_data = {
    'captions': '["A cat playing"]'
}

response = requests.post(url, files=files, data=form_data)