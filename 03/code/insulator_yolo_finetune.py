from ultralytics import YOLO


model_path = '/content/drive/MyDrive/YOLO_Project/runs/glass_porcelain_detect/weights/best.pt'
model = YOLO(model_path)


model.train(
    data='/content/drive/MyDrive/YOLO_Project/data.yaml',
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
    
    
    project='/content/drive/MyDrive/YOLO_Project/runs', 
    name='insulator_finetune_50epochs' 
)
