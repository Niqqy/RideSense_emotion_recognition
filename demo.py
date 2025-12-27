# demo.py - Real-Time Emotion Recognition Demo
# Can be used with webcam, video files, or images

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# ========== CONFIGURATION ==========
MODEL_PATH = 'models/best_model.keras'
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Colors for each emotion (BGR format)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 0),    # Dark Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 0),      # Green
    'sad': (255, 0, 0),        # Blue
    'surprise': (0, 255, 255), # Yellow
    'neutral': (128, 128, 128) # Gray
}

# ========== LOAD MODEL ==========
print("Loading emotion recognition model...")
model = load_model(MODEL_PATH)
print("✓ Model loaded successfully!")

# Load face detector (Haar Cascade - comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✓ Face detector loaded!")

# ========== PREPROCESSING FUNCTION ==========
def preprocess_face(face_img):
    """Preprocess face image for model prediction"""
    # Resize to 48x48
    face_img = cv2.resize(face_img, (48, 48))
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Normalize pixels to [0, 1]
    face_img = face_img / 255.0
    # Reshape for model input
    face_img = face_img.reshape(1, 48, 48, 1)
    return face_img

# ========== STATE MAPPING (Section 3.4.4 from your methodology) ==========
def map_to_safety_state(emotion, confidence, is_driver=True):
    """Map emotion to safety state as per RideSense system"""
    if is_driver:
        # Driver Safety States
        if emotion in ['happy', 'neutral'] and confidence > 0.5:
            return 'FIT', (0, 255, 0)  # Green
        elif emotion in ['sad', 'neutral'] and confidence > 0.6:
            return 'DROWSY', (0, 165, 255)  # Orange
        elif emotion in ['angry', 'disgust', 'fear', 'sad', 'surprise']:
            return 'UNFIT', (0, 0, 255)  # Red
        else:
            return 'FIT', (0, 255, 0)
    else:
        # Passenger Safety States
        if emotion == 'happy' and confidence > 0.5:
            return 'ENJOYING', (0, 255, 0)  # Green
        elif emotion == 'sad':
            return 'NOT ENJOYING', (0, 0, 255)  # Red
        else:
            return 'NEUTRAL', (128, 128, 128)  # Gray

# ========== EMOTION DETECTION FUNCTION ==========
def detect_emotions(frame, show_probabilities=True):
    """Detect faces and emotions in a frame"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess face
        processed_face = preprocess_face(face_roi)
        
        # Predict emotion
        predictions = model.predict(processed_face, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Get safety state (assuming driver for demo)
        safety_state, state_color = map_to_safety_state(emotion, confidence, is_driver=True)
        
        # Draw rectangle around face
        color = EMOTION_COLORS[emotion]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display emotion label
        label = f"{emotion.upper()} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display safety state
        cv2.putText(frame, f"State: {safety_state}", (x, y+h+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Show probability distribution if enabled
        if show_probabilities:
            prob_y = y + 50
            for i, (label_name, prob) in enumerate(zip(EMOTION_LABELS, predictions)):
                bar_length = int(prob * 150)
                cv2.rectangle(frame, (x+w+10, prob_y), (x+w+10+bar_length, prob_y+15), 
                            EMOTION_COLORS[label_name], -1)
                cv2.putText(frame, f"{label_name}: {prob*100:.0f}%", 
                          (x+w+170, prob_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                prob_y += 20
    
    return frame, len(faces)

# ========== MODE 1: WEBCAM DEMO ==========
def webcam_demo():
    """Run real-time emotion detection on webcam"""
    print("\n========== WEBCAM MODE ==========")
    print("Press 'q' to quit")
    print("Press 'p' to toggle probability display")
    print("Press 's' to save screenshot")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW is important for Windows
        if cap.isOpened():
            print(f"✓ Camera {camera_index} opened successfully!")
            break
        cap.release()
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    show_probs = True
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, num_faces = detect_emotions(frame, show_probabilities=show_probs)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        # Display FPS and face count
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Faces: {num_faces}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('RideSense - Emotion Recognition', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_probs = not show_probs
            print(f"Probability display: {'ON' if show_probs else 'OFF'}")
        elif key == ord('s'):
            filename = f'screenshot_{int(time.time())}.jpg'
            cv2.imwrite(filename, processed_frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam demo ended")

# ========== MODE 2: VIDEO FILE DEMO ==========
def video_demo(video_path):
    """Run emotion detection on a video file"""
    print(f"\n========== VIDEO MODE ==========")
    print(f"Processing: {video_path}")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")
    
    # Optional: Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Process frame
        processed_frame, num_faces = detect_emotions(frame, show_probabilities=False)
        
        # Display progress
        cv2.putText(processed_frame, f"Frame: {frame_num}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write to output video
        out.write(processed_frame)
        
        # Show frame
        cv2.imshow('RideSense - Video Processing', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Processed video saved as: output_video.mp4")

# ========== MODE 3: IMAGE DEMO ==========
def image_demo(image_path):
    """Run emotion detection on a single image"""
    print(f"\n========== IMAGE MODE ==========")
    print(f"Processing: {image_path}")
    
    # Read image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image: {image_path}")
        return
    
    # Process image
    processed_frame, num_faces = detect_emotions(frame, show_probabilities=True)
    
    # Save result
    output_path = 'output_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, processed_frame)
    print(f"✓ Result saved as: {output_path}")
    
    # Display result
    cv2.imshow('RideSense - Image Result', processed_frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== MAIN MENU ==========
def main():
    print("\n" + "="*60)
    print("      RIDESENSE EMOTION RECOGNITION SYSTEM")
    print("="*60)
    print("\nSelect Demo Mode:")
    print("1. Webcam (Real-time)")
    print("2. Video File")
    print("3. Image File")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        webcam_demo()
    elif choice == '2':
        video_path = input("Enter video file path: ")
        video_demo(video_path)
    elif choice == '3':
        image_path = input("Enter image file path: ")
        image_demo(image_path)
    elif choice == '4':
        print("Goodbye!")
        return
    else:
        print("Invalid choice!")
    
    # Ask if user wants to continue
    again = input("\nRun another demo? (y/n): ")
    if again.lower() == 'y':
        main()

# ========== RUN ==========
if __name__ == "__main__":
    main()