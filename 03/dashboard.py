import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="UAV Insulator Inspection", layout="wide")
st.title("🛰️ UAV Insulator Classification")
st.write("Upload a photo from the drone to detect: Porcelain, Glass, or Composite")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

st.sidebar.header("Controls")
conf_val = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

uploaded_file = st.file_uploader("Upload UAV Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Run the Model
    results = model.predict(source=img, conf=conf_val)
    
    # Show results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Objects")
        st.image(results[0].plot(), use_container_width=True)
    
    with col2:
        st.subheader("Summary")
        labels = [model.names[int(c)] for c in results[0].boxes.cls]
        if labels:
            for label in set(labels):
                st.write(f"- **{label.capitalize()}**: {labels.count(label)} found")
        else:
            st.write("No insulators detected.")
