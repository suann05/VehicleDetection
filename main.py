import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from fpdf import FPDF
import tempfile
import os
from streamlit_js_eval import streamlit_js_eval

# Load YOLOv5 model
# model = yolov5.load('yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Streamlit UI Layout
st.set_page_config(layout="wide")
st.title("Vehicle Detection System")

# Initialize session state for reset
if "detected_results" not in st.session_state:
    st.session_state.detected_results = None
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None
if "categorized_count" not in st.session_state:
    st.session_state.categorized_count = None

# Sidebar - Left Panel
with st.sidebar:
    st.subheader("Upload Media:")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"], key="file_uploader")
    detect_button = st.button("Detect")
    reset_button = st.button("Reset")

# Reset Logic
if reset_button:
    st.session_state.detected_results = None
    st.session_state.annotated_image = None
    st.session_state.categorized_count = None
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    st.stop()  # Stop the execution to allow the refresh

if uploaded_file:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detection Logic
    if detect_button:
        results = model(image_cv)  # Pass the image to the YOLOv5 model

        # Get the annotated image
        annotated_image = np.squeeze(results.render())  # results.render() modifies the image in place
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display

        # Store detection results in session state
        st.session_state.detected_results = results
        st.session_state.annotated_image = annotated_image

        # Filter detected vehicle types (common vehicle class labels)
        vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train']
        detected_classes = [model.names[int(x[5])] for x in results.xyxy[0]]

        # Only count vehicle-related detections
        vehicle_detected = [cls for cls in detected_classes if cls in vehicle_classes]
        st.session_state.categorized_count = {cls: vehicle_detected.count(cls) for cls in set(vehicle_detected)}

# Display detection results if available
if st.session_state.annotated_image is not None:
    st.image(st.session_state.annotated_image, caption="Detected Vehicles", use_container_width=True)

    st.subheader("Results:")
    for cls, count in st.session_state.categorized_count.items():
        st.metric(f"{cls.capitalize()}s", count)

    if st.button("Export PDF"):
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Vehicle Detection Results", ln=True, align='C')

        # Save the annotated image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            img = Image.fromarray(st.session_state.annotated_image)
            img.save(temp_image_file.name, format="JPEG")

            # Add image to the PDF
            pdf.image(temp_image_file.name, x=10, y=30, w=180)

        # Add some space before the results section
        pdf.ln(130)  # Adds some space between the image and text

        # Add Results to PDF dynamically
        for cls, count in st.session_state.categorized_count.items():
            pdf.cell(0, 10, txt=f"Total {cls.capitalize()}s: {count}")
            pdf.ln(10)

        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
            pdf.output(temp_pdf_file.name)

            # Provide download button for the PDF file
            st.download_button(
                label="Download PDF",
                data=open(temp_pdf_file.name, "rb").read(),
                file_name="detection_results.pdf",
                mime="application/pdf",
            )
