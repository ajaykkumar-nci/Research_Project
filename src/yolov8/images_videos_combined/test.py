from ultralytics import YOLO
import os
import cv2
from shared_config import MODEL_PATH, TEST_DATA_PATH, YAML_PATH, UNIFIED_RESULTS_PATH, get_image_result_path

model = YOLO(MODEL_PATH)

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

for img_name in os.listdir(TEST_DATA_PATH):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(TEST_DATA_PATH, img_name)
        img_result_path = get_image_result_path(img_name)
        
        results = model(img_path)
        img = cv2.imread(img_path)
        img_with_boxes = draw_boxes(img, results)
        cv2.imwrite(os.path.join(img_result_path, f"yolo_result_{img_name}"), img_with_boxes)
        
        with open(os.path.join(img_result_path, "yolo_detections.txt"), "w") as f:
            f.write(f"YOLO Detections for {img_name}:\n")
            for r in results:
                for c in r.boxes.cls:
                    f.write(f"  - {model.names[int(c)]}\n")

val_results = model.val(data=YAML_PATH)

with open(os.path.join(UNIFIED_RESULTS_PATH, "yolo_validation_results.txt"), "w") as f:
    f.write("YOLO Validation Results:\n")
    f.write(f"mAP50: {val_results.box.map50:.3f}\n")
    f.write(f"mAP50-95: {val_results.box.map:.3f}\n")

print(f"YOLO testing completed. Results saved in {UNIFIED_RESULTS_PATH}")
