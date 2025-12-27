# Install required packages (run in Colab)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Check your data structure
import os

train_path = r'C:\Users\Administrator\Downloads\archive\train'
print("Classes found:")
for class_name in os.listdir(train_path):
    class_path = os.path.join(train_path, class_name)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        print(f"{class_name}: {num_images} images")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.1  # Reserve 10% for validation
)

# Training data (90% of train folder)
train_data = train_datagen.flow_from_directory(
    r'C:\Users\Administrator\Downloads\archive\train',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='training',  # This is the training subset
    shuffle=True
)

# Validation data (10% of train folder)
validation_data = train_datagen.flow_from_directory(
    r'C:\Users\Administrator\Downloads\archive\train',  # Same folder!
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    subset='validation',  # This is the validation subset
    shuffle=False
)

# Test data (separate folder)
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    r'C:\Users\Administrator\Downloads\archive\test',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)


print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {validation_data.samples}")
print(f"Test samples: {test_data.samples}")
print(f"Class indices: {train_data.class_indices}")

# ==================== PHASE 4: BUILD MODEL ====================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def build_emotion_model():
    model = Sequential()
    
    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                     kernel_initializer='he_normal', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', 
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', 
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Convolutional Block 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', 
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', 
                     kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(7, activation='softmax'))
    
    return model

model = build_emotion_model()
model.summary()

# ==================== PHASE 5: COMPILE MODEL ====================
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== PHASE 6: SETUP CALLBACKS ====================
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import os

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=0.00001
)

tensorboard = TensorBoard(log_dir='logs')

callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]

# ==================== PHASE 7: TRAIN MODEL ====================
steps_per_epoch = train_data.samples // train_data.batch_size
validation_steps = validation_data.samples // validation_data.batch_size

print("\n========== STARTING TRAINING ==========")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=validation_data,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

print("\n========== TRAINING COMPLETE ==========")