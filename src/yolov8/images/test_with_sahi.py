from ultralytics import YOLO
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image, visualize_object_predictions
from PIL import Image
import numpy as np
import cv2
from shared_config import MODEL_PATH, TEST_DATA_PATH, UNIFIED_RESULTS_PATH, get_image_result_path

yolo_model = YOLO(MODEL_PATH)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=MODEL_PATH,
    confidence_threshold=0.3,
    device="cuda:0"
)

def process_image(image_path):
    img_name = os.path.basename(image_path)
    img_result_path = get_image_result_path(img_name)
    
    image = read_image(image_path)
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Save SAHI visualization
    sahi_vis = visualize_object_predictions(
        image=image,
        object_prediction_list=result.object_prediction_list,
    )
    
    # Handle the case where sahi_vis is a dictionary
    if isinstance(sahi_vis, dict):
        # Assuming the dictionary contains the image data
        # You might need to adjust this based on the actual structure of the dictionary
        if 'image' in sahi_vis:
            image_data = sahi_vis['image']
        else:
            print(f"Unexpected dictionary structure for {img_name}. Keys: {sahi_vis.keys()}")
            return
    elif isinstance(sahi_vis, np.ndarray):
        image_data = sahi_vis
    else:
        print(f"Unexpected type for visualization: {type(sahi_vis)} for {img_name}")
        return

    # Save the image using OpenCV
    cv2.imwrite(os.path.join(img_result_path, f"sahi_result_{img_name}"), image_data)

    with open(os.path.join(img_result_path, "sahi_detections.txt"), "w") as f:
        f.write(f"SAHI Detections for {img_name}:\n")
        for prediction in result.object_prediction_list:
            f.write(f"  - Category: {prediction.category.name}, Score: {prediction.score.value:.2f}\n")

# Process all images in the test directory
for filename in os.listdir(TEST_DATA_PATH):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(TEST_DATA_PATH, filename)
        process_image(image_path)

print(f"SAHI testing completed. Results saved in {UNIFIED_RESULTS_PATH}")
