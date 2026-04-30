import os
from ultralytics import YOLO

BASE_DIR = os.getcwd()

data_path = os.path.join(BASE_DIR, 'code', 'data.yaml')

model = YOLO('yolov8n.pt') 

results = model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    workers=6,
    batch=16
)
