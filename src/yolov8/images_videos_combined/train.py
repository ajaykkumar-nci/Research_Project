from ultralytics import YOLO
import os

model = YOLO("yolov8s.pt")
cfg_path = "/home/ubuntu/Research_Project/yolo_models/cfg"
output_path = "/home/ubuntu/Research_Project/yolo_models/result/yolov8/combined"

results = model.train(data=os.path.join(cfg_path,"Visdrone_combined.yaml"), epochs=50, imgsz=640, patience = 5, batch=32, , project=output_path)
