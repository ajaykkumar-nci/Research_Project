from ultralytics import YOLO
import os

model = YOLO("yolov8s.pt")
cfg_path = "../cfg"
output_path = "../result/yolov8"

results = model.train(data=os.path.join(cfg_path,"Visdrone.yaml"), 
                        epochs=200, 
                        imgsz=800, 
                        close_mosaic=10,
                        shear=0.1,
                        bgr=0.1, 
                        mixup=0.1,
                        copy_paste=0.1, 
                        patience=100, 
                        cos_lr=True,
                        dropout = 0.5, 
                        batch=16, 
                        project=output_path)
