import os

MODEL_PATH = "/home/ubuntu/Research_Project/object_detect/runs/detect/train7/weights/best.pt"
TEST_DATA_PATH = "/home/ubuntu/Research_Project/Visdrone_combined/test/images"
YAML_PATH = "/home/ubuntu/Research_Project/yolo_models/cfg/Visdrone_combined.yaml"
UNIFIED_RESULTS_PATH = "/home/ubuntu/Research_Project/yolo_models/output/yolov8/combined"

# Ensure the unified results directory exists
os.makedirs(UNIFIED_RESULTS_PATH, exist_ok=True)

def get_image_result_path(img_name):
    base_name = os.path.splitext(img_name)[0]
    img_result_path = os.path.join(UNIFIED_RESULTS_PATH, base_name)
    os.makedirs(img_result_path, exist_ok=True)
    return img_result_path