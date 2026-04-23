import streamlit as st
import cv2
import numpy as np
from PIL import Image
# from src.predict import MaskPredictor
import sys
import os
sys.path.append(os.path.abspath("src"))

from predict import MaskPredictor


# Set Streamlit page configurations
st.set_page_config(
    page_title="Mask Detection & Face Safety Compliance Suite",
    page_icon="😷",
    layout="wide"
)

# Initialize the inference engine globally
@st.cache_resource
def load_engine():
    return MaskPredictor("models/mask_detector.h5")

predictor = load_engine()

# --- CSS Styling for Premium Aesthetics ---
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    .status-safe { color: #2ea043; font-weight: bold; }
    .status-violation { color: #f85149; font-weight: bold; }
    .status-uncertain { color: #d29922; font-weight: bold; }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Overview
st.title("🛡️ Mask Detection & Safety Compliance Tracker")
st.markdown("""
Welcome to the Real-Time Facial Compliance Checker! 
This Computer Vision system evaluates mask-wearing protocol using **MobileNetV2**.
""")

# Sidebar Navigation
st.sidebar.title("Configuration / Navigation")
app_mode = st.sidebar.selectbox("Choose the Operation Mode", ["Image Upload Detection", "Live Real-Time Webcam", "Model Analytics"])

if app_mode == "Image Upload Detection":
    st.header("📸 Static Image Inference")
    st.markdown("Upload a crowd or individual image (jpg, jpeg, png).")
    uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert PIL image to OpenCV format (BGR)
        image_np = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Running Inference Engine..."):
            processed_bgr = predictor.process_frame(image_bgr)
            
        # Convert BGR back to RGB for display in Streamlit
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        
        st.image(processed_rgb, caption="Processed Analysis Output", use_column_width=True)
        st.success("Analysis Complete!")

elif app_mode == "Live Real-Time Webcam":
    st.header("🔴 Live Monitoring Feed")
    st.markdown("Initiating Real-time Deep Learning prediction loop.")
    
    start_cam = st.button("Start Camera Feed")
    stop_cam = st.button("Stop Camera Feed")
    
    FRAME_WINDOW = st.image([])
    
    if start_cam:
        # cv2.VideoCapture(0) loads the default camera driver
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access the webcam.")
        else:
            st.success("Webcam Activated!")
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to fetch frame from camera")
                break
                
            # OpenCV provides BGR natively, apply inference:
            annotated_frame = predictor.process_frame(frame)
            
            # Convert back to RGB for Streamlit frontend
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)
            
            # Streamlit logic doesn't cleanly break inner loops from other buttons
            # We break using standard streamlit Session State or user reload
            if stop_cam:
                break
                
        cap.release()

elif app_mode == "Model Analytics":
    st.header("📊 Model Metrics & Confusion Matrix")
    st.markdown("Explore the Deep Learning Training history and class-based Validation.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training/Validation Graph")
        try:
            st.image("models/training_history.png", use_column_width=True)
        except:
            st.warning("Training plots not found. Please run the generation script.")
            
    with col2:
        st.subheader("Confusion Matrix")
        try:
            st.image("models/confusion_matrix.png", use_column_width=True)
        except:
            st.warning("Confusion matrix not found.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ | Keras + Streamlit")
