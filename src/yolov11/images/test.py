from ultralytics import YOLO
import os
import cv2

model_path = "/home/ubuntu/Research_Project/object_detect/runs/detect/train7/weights/best.pt"
model = YOLO(model_path)

# Path to your test dataset
test_data_path = "/home/ubuntu/Research_Project/Visdrone_images/test/images"
yaml_path = "/home/ubuntu/Research_Project/yolo_models/cfg/Visdrone.yaml"

# Path to save results
results_path = "./test_results"
os.makedirs(results_path, exist_ok=True)

# Validate that the paths exist
print(f"Model path exists: {os.path.exists(model_path)}")
print(f"YAML path exists: {os.path.exists(yaml_path)}")
print(f"Test data path exists: {os.path.exists(test_data_path)}")

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(image, f"{model.names[int(c)]}", (int(b[0]), int(b[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Test the model on images in the test dataset
for img_name in os.listdir(test_data_path):
    if img_name.lower().endswith(('.jpg')):
        img_path = os.path.join(test_data_path, img_name)
        results = model(img_path)
        img = cv2.imread(img_path)
        img_with_boxes = draw_boxes(img, results)
        cv2.imwrite(os.path.join(results_path, f"result_{img_name}"), img_with_boxes)
        print(f"Detections for {img_name}:")
        for r in results:
            for c in r.boxes.cls:
                print(f"  - {model.names[int(c)]}")
        print()

val_results = model.val(data=yaml_path)

print("Validation Results:")
print(f"mAP50: {val_results.box.map50:.3f}")
print(f"mAP50-95: {val_results.box.map:.3f}")

print(f"Test completed. Results saved in {results_path}")
