results = model.train(
    data='/content/drive/MyDrive/YOLO_Project/data.yaml',
    epochs=100,
    imgsz=640,
    workers=6,
    batch=16
)