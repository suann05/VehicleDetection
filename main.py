import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

model = load_model()

# Streamlit app UI
st.title("Vehicle Detection and Counting App")
st.text("Upload an image to detect and count vehicles")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Load and display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert to OpenCV format (BGR for YOLOv5 compatibility)
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Perform detection
        results = model(image_cv)

        # Annotate and display the results
        annotated_image = np.array(results.render()[0])  # Render and retrieve the annotated image
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB
        st.image(annotated_image, caption="Detected Vehicles", use_container_width=True)

        # Display vehicle count
        vehicle_count = len(results.xyxy[0])  # Each detection is a row in the tensor
        st.write(f"Total Vehicles Detected: {vehicle_count}")

        # Categorize by type (e.g., car, truck)
        detected_classes = [model.names[int(x[5])] for x in results.xyxy[0]]
        categorized_count = {cls: detected_classes.count(cls) for cls in set(detected_classes)}
        st.write("Categorized Count:", categorized_count)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload an image to proceed.")
