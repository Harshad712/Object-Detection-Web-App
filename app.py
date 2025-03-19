import gradio as gr
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json

# Load your trained YOLOv10 model
model_path = "./best.pt"  # Adjust path if needed
model = YOLO(model_path)

# Load metrics data
with open("./metrics.json", "r") as f:
    metrics = json.load(f)

# Define the prediction function
def predict(image):
    results = model.predict(image)

    # Extract bounding boxes, classes, and confidence
    boxes = results[0].boxes.data
    annotated_image = np.array(image)

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        class_name = model.names[int(cls)]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to PIL Image
    annotated_image = Image.fromarray(annotated_image)

    # Create a detection results table
    result_data = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        class_name = model.names[int(cls)]
        result_data.append([class_name, f"{conf:.2f}"])

    # Metrics Table
    metrics_table = [
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["mAP@50", f"{metrics['map50']:.4f}"],
        ["mAP@50-95", f"{metrics['map50_95']:.4f}"]
    ]

    # Class-wise Table
    class_wise_table = [
        ["Class", "mAP"],
        ["WBC", f"{metrics['class_wise_metrics']['WBC']['map']:.4f}"],
        ["RBC", f"{metrics['class_wise_metrics']['RBC']['map']:.4f}"],
        ["Platelets", f"{metrics['class_wise_metrics']['Platelets']['map']:.4f}"]
    ]

    return annotated_image, result_data, metrics_table, class_wise_table

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Detected Objects"),
        gr.Dataframe(headers=["Class", "Confidence"], label="Detection Results"),
        gr.Dataframe(headers=["Metric", "Score"], label="Overall Metrics"),
        gr.Dataframe(headers=["Class", "mAP"], label="Class-wise Metrics")
    ],
    title="Object Detection Web App",
    description="Upload an image and the model will detect WBC, RBC, or Platelets with confidence scores, precision, and recall."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
