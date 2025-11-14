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
