# streamlit_app.py - Web Interface for RideSense Emotion Recognition

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import time

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="RideSense Emotion Recognition",
    page_icon="🚗",
    layout="wide"
)

# ========== CONFIGURATION ==========
MODEL_PATH = 'models/best_model.keras'
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTION_COLORS = {
    'angry': (255, 0, 0),      # Red (RGB for display)
    'disgust': (0, 128, 0),    
    'fear': (128, 0, 128),     
    'happy': (0, 255, 0),      # Green
    'sad': (0, 0, 255),        # Blue
    'surprise': (255, 255, 0), # Yellow
    'neutral': (128, 128, 128) 
}

# ========== LOAD MODEL (cached) ==========
@st.cache_resource
def load_emotion_model():
    """Load model once and cache it"""
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_emotion_model()

# ========== PREPROCESSING ==========
def preprocess_face(face_img):
    """Preprocess face for model prediction"""
    face_img = cv2.resize(face_img, (48, 48))
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, 48, 48, 1)
    return face_img

# ========== STATE MAPPING ==========
def map_to_safety_state(emotion, confidence):
    """Map emotion to safety state"""
    if emotion in ['happy', 'neutral'] and confidence > 0.5:
        return 'FIT', '🟢'
    elif emotion in ['sad', 'neutral'] and confidence > 0.6:
        return 'DROWSY', '🟡'
    elif emotion in ['angry', 'disgust', 'fear', 'sad', 'surprise']:
        return 'UNFIT', '🔴'
    else:
        return 'FIT', '🟢'

# ========== EMOTION DETECTION ==========
def detect_emotions_on_image(image):
    """Detect emotions in an image"""
    # Convert PIL to OpenCV
    img_array = np.array(image)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_roi = gray[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        
        # Predict
        predictions = model.predict(processed_face, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Get safety state
        safety_state, emoji = map_to_safety_state(emotion, confidence)
        
        # Draw on frame
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, f"{emotion.upper()} ({confidence*100:.1f}%)", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"State: {safety_state}", 
                   (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Store results
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'safety_state': safety_state,
            'emoji': emoji,
            'predictions': predictions
        })
    
    # Convert back to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, results

# ========== STREAMLIT UI ==========
st.title("🚗 RideSense: Emotion Recognition System")
st.markdown("### Real-time Emotion Detection for Transportation Safety")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.info("""
    **RideSense** uses deep learning to detect emotions and assess driver/passenger safety states.
    
    **Emotions Detected:**
    - 😠 Angry
    - 🤢 Disgust  
    - 😨 Fear
    - 😊 Happy
    - 😢 Sad
    - 😲 Surprise
    - 😐 Neutral
    
    **Safety States:**
    - 🟢 FIT: Safe to drive
    - 🟡 DROWSY: Warning
    - 🔴 UNFIT: Unsafe
    """)
    
    st.header("⚙️ Model Info")
    st.write(f"**Accuracy:** 64.0%")
    st.write(f"**Dataset:** FER2013")
    st.write(f"**Images:** 28,709 training")

# Main content
tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "🎥 Webcam (Live)", "📊 Model Performance"])

# ========== TAB 1: IMAGE UPLOAD ==========
with tab1:
    st.header("Upload an Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            with st.spinner("Analyzing emotions..."):
                # Process image
                processed_image, results = detect_emotions_on_image(image)
                st.image(processed_image, use_container_width=True)
        
        # Display results
        if results:
            st.success(f"✅ Detected {len(results)} face(s)")
            
            for i, result in enumerate(results):
                st.markdown(f"### Face {i+1}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Emotion", result['emotion'].upper())
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                
                with col3:
                    st.metric("Safety State", f"{result['emoji']} {result['safety_state']}")
                
                # Probability chart
                st.subheader("Emotion Probabilities")
                prob_df = {
                    'Emotion': EMOTION_LABELS,
                    'Probability': result['predictions']
                }
                st.bar_chart(prob_df, x='Emotion', y='Probability')
        else:
            st.warning("⚠️ No faces detected in the image")

# ========== TAB 2: WEBCAM ==========
with tab2:
    st.header("Live Webcam Detection")
    st.info("📹 Click 'Start Webcam' to begin real-time emotion detection")
    
    # Webcam input
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    
    FRAME_WINDOW = st.image([])
    
    if start_button:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam")
        else:
            st.success("✅ Webcam started!")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect emotions
                processed_frame, results = detect_emotions_on_image(Image.fromarray(frame_rgb))
                
                # Display
                FRAME_WINDOW.image(processed_frame)
                
                # Stop if button clicked (Note: this is a limitation of Streamlit)
                if stop_button:
                    break
                
                time.sleep(0.03)  # ~30 FPS
            
            cap.release()
            st.info("Webcam stopped")

# ========== TAB 3: MODEL PERFORMANCE ==========
with tab3:
    st.header("Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Accuracy", "64.0%")
    
    with col2:
        st.metric("Weighted F1-Score", "62.8%")
    
    with col3:
        st.metric("Macro Precision", "63.6%")
    
    st.subheader("📊 Confusion Matrix")
    
    # Check if confusion matrix image exists
    import os
    if os.path.exists('confusion_matrix_percentage.png'):
        st.image('confusion_matrix_percentage.png', use_container_width=True)
    else:
        st.warning("Run evaluate.py first to generate confusion matrix")
    
    st.subheader("📈 Per-Class Metrics")
    
    # Load metrics if available
    if os.path.exists('per_class_metrics.csv'):
        import pandas as pd
        metrics_df = pd.read_csv('per_class_metrics.csv')
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("Run evaluate.py first to generate metrics")
    
    st.subheader("🔄 Top Misclassifications")
    
    if os.path.exists('misclassifications.csv'):
        import pandas as pd
        misclass_df = pd.read_csv('misclassifications.csv')
        st.dataframe(misclass_df.head(10), use_container_width=True)
    else:
        st.warning("Run evaluate.py first to generate misclassifications")

# Footer
st.markdown("---")
st.markdown("**RideSense Emotion Recognition System** | Built with TensorFlow & Streamlit")