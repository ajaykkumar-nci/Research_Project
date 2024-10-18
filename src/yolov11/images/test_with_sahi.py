from ultralytics import YOLO
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

model_path = "/home/ubuntu/Research_Project/object_detect/runs/detect/train5/weights/best.pt"
yolo_model = YOLO(model_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0"
)

def process_image(image_path):
    image = read_image(image_path)
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    export_dir = os.path.join("sahi_results", base_filename)
    os.makedirs(export_dir, exist_ok=True)
    result.export_visuals(export_dir=export_dir)
    print(f"Found {len(result.object_prediction_list)} objects in {image_path}")
    for prediction in result.object_prediction_list:
        print(f"Category: {prediction.category.name}, Score: {prediction.score.value}")

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory_path, filename)
            process_image(image_path)

os.makedirs("sahi_results", exist_ok=True)
process_directory("/home/ubuntu/Research_Project/Visdrone_images/test/images")

print("SAHI processing completed. Check the 'sahi_results' directory for visualizations.")
