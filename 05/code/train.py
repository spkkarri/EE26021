from ultralytics import YOLO

# load model
model = YOLO("yolov8n.pt")  # lightweight model

# train model
model.train(
    data="data/data.yaml",
    epochs=10,
    imgsz=640
)