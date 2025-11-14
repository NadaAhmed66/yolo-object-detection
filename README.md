# 🧠 YOLO Object Detection App

Real-Time Object Detection using **YOLOv8** + **Streamlit**

## 📌 Overview

This project is a simple and interactive web application for **Object Detection** using **YOLOv8**.
You can upload any image and the app will detect objects, draw bounding boxes, and display class labels with confidence scores.

The interface is built with **Streamlit**, and the detection model is loaded using the **Ultralytics YOLO** library.

---

## 🚀 Features

* Upload images (JPG, PNG, JPEG)
* Detect multiple objects in the image
* Filter detected objects using a multi-select dropdown
* Display bounding boxes and class labels
* Fast and lightweight (uses YOLOv8n)

---

## 🛠️ Technologies Used

* **Python**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **NumPy**
* **Streamlit**
* **Pillow (PIL)**

---

## 📁 Project Structure

```
├── object_detection.py
├── README.md
├── requirements.txt  (optional)
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/yolo-object-detection.git
cd yolo-object-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
ultralytics
streamlit
opencv-python
numpy
Pillow
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run object_detection.py
```

---

## 🧩 Code

```python
%%writefile object_detection.py
from ultralytics import YOLO
import streamlit as st 
import cv2
import numpy as np 
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model=load_model()
st.title('object detection ')
upload=st.file_uploader("upload an image.......",type=['png','jpg','jpeg'])

if upload is not None:
    img=Image.open(upload)
    img_array=np.array(img)

    results=model(img_array)[0]
    boxes=results.boxes

    class_id=boxes.cls.cpu().numpy().astype(int)
    confidence=boxes.conf.cpu().numpy()
    xyxy=boxes.xyxy.cpu().numpy().astype(int)

    class_name=[model.names[c] for c in class_id]

    unique_classes=sorted(set(class_name))
    select_classes=st.multiselect("image......",unique_classes,default=unique_classes)

    for box,cls_name,conf in zip(xyxy,class_name,confidence):
        if cls_name in select_classes:
            x1,y1,x2,y2=box
            label=f"{cls_name} {conf:.2f}"

            cv2.rectangle(img_array,(x1,y1),(x2,y2),(22,100,50),2)
            cv2.putText(img_array,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,100,50),2)

    st.image(img_array,use_container_width=True,caption='detected object')
```

---

## 🖼️ Example Output

The application will display the uploaded image with detected objects highlighted.

---

## 🙌 Author

**Nod — AI Developer**

If you like this repo, don’t forget to ⭐ star it!

---
