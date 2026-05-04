import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, 'data.yaml')

model = YOLO('yolov8n.pt')

results = model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    workers=6,
    batch=16,
    
    project=os.path.join(BASE_DIR, '..', 'runs'),
    name='glass_porcelain_detect' 
)
