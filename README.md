# 🚗 RideSense: Emotion Recognition System

Deep Learning-based real-time emotion recognition for enhanced transportation safety in Kenya.

## 📊 Model Performance
- **Test Accuracy**: 64.0%
- **Weighted F1-Score**: 62.8%
- **Dataset**: FER2013 (28,709 training images)
- **Emotions Detected**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 🚀 Features
- Real-time webcam emotion detection
- Video file processing
- Image analysis
- Safety state mapping (FIT/UNFIT/DROWSY)
- Interactive web interface with Streamlit

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/Niqay/RideSense_emotion_recognition.git
cd RideSense_emotion_recognition

# Install dependencies
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn streamlit
```

## 📁 Project Structure
```
├── tester.py              # Model training script
├── evaluation.py          # Model evaluation and metrics
├── demo.py                # Real-time demo (webcam/video/image)
├── streamlit_app.py       # Web interface
├── test_camera.py         # Camera testing utility
└── README.md              # Documentation
```

## 🎯 Usage

### Train the Model
```bash
python tester.py
```

### Evaluate the Model
```bash
python evaluation.py
```

### Run Real-time Demo
```bash
python demo.py
# Choose option 1 for webcam, 2 for video, 3 for image
```

### Run Web Interface
```bash
streamlit run streamlit_app.py
```

## 📈 Results

### Per-Emotion Performance
The model achieves the following performance across emotion classes:
- Best performing: Happy and Neutral emotions
- Most challenging: Fear and Disgust (often confused with similar emotions)

### Safety State Mapping
- **FIT**: Happy or Neutral (confidence > 50%) - Driver is safe to operate vehicle
- **DROWSY**: Sad or Neutral (confidence > 60%) - Warning state, requires attention
- **UNFIT**: Angry, Disgust, Fear, Sad, or Surprise - Driver should not operate vehicle

## 🔬 Methodology
Built using Convolutional Neural Networks (CNN) with:
- 3 Convolutional blocks with progressive filter increase (32 → 64 → 128)
- Batch normalization for training stability
- Dropout regularization (25% for conv layers, 50% for dense layers)
- Data augmentation (rotation, shifts, flipping, zoom)
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting

## 📝 Notes
- The trained model file (`best_model.keras`) is not included in this repository due to GitHub size constraints (15-20MB)
- You need to train the model first using `tester.py` before running demos
- Dataset should be organized in the following structure:
```
  data/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── sad/
  │   ├── surprise/
  │   └── neutral/
  └── test/
      └── (same structure)
```

## 🎓 Academic Context
This project was developed as part of a computer vision course focused on emotion recognition systems for transportation safety applications in Kenya.

## 👨‍💻 Author
Developed by Nicolette Nkirote

## 📄 License
This project is available for academic and educational purposes.
