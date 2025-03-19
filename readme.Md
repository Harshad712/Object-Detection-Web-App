# Object Detection Web App

This project is a web-based object detection application that uses a fine-tuned YOLOv10 model to detect WBC, RBC, and Platelets in medical images. The application is built using **Gradio** and deployed on **Hugging Face Spaces**.

---

##  Features
 Upload an image for object detection.  
 Displays detected objects with **bounding boxes** and **confidence scores**.  
 Shows a **Precision**, **Recall**, and **mAP** table for performance metrics.  
 Clean and user-friendly interface built using Gradio.  

---

##  Project Structure
```
├── app.py                # Gradio web app code
├── best.pt               # Fine-tuned YOLOv10 model weights
├── metrics.json          # Precision, Recall, and mAP metrics file
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

##  How to Run Locally

1. **Clone the Repository**
```bash
git clone github.com/Harshad712/Object-Detection-Web-App
cd object-detection-app
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the Model File**
Ensure `best.pt` (your YOLOv10 model file) is in the project folder. If missing, download it and place it in the correct location.

4. **Run the Application**
```bash
python app.py
```

5. **Open the Web App**
Once the app starts , Gradio will generate a URL like:
```
Running on local URL: http://localhost:7860
```
Click the link to interact with the web app.

---

##  Deployment on Hugging Face Spaces
The app is deployed on Hugging Face Spaces for easy accessibility.

🔗 **[Live Demo on Hugging Face](https://huggingface.co/spaces/Harshad712/object-detection-app)**

---

##  Model Performance Metrics
| Metric                | Value |
|----------------------|--------|
| Precision (B)         | 0.8402 |
| Recall (B)            | 0.8635 |
| mAP50 (B)             | 0.9204 |
| mAP50-95 (B)          | 0.6585 |

---

##  Requirements
Ensure the following dependencies are listed in your `requirements.txt` file :
```
gradio
torch
ultralytics
opencv-python
pillow
```

---




