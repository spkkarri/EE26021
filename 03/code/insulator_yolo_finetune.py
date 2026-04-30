import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '..', 'runs', 'glass_porcelain_detect', 'weights', 'best.pt')
model = YOLO(model_path)

data_config = os.path.join(BASE_DIR, 'data.yaml')

model.train(
    data=data_config,
    epochs=50,                  
    imgsz=1024,                 
    batch=16,                   
    amp=True,                   
    
    
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,                  
    copy_paste=0.2,             
    mosaic=1.0,
    mixup=0.15,
    erasing=0.4,                
    
    
    lr0=0.0005,               
    optimizer='AdamW',
    
    
    project=os.path.join(BASE_DIR, '..', 'runs'), 
    name='insulator_finetune_50epochs' 
)
