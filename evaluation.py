# evaluate.py - Model Evaluation Script

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

print("========== LOADING TRAINED MODEL ==========")

# Load the best saved model
model = load_model('models/best_model.keras')
print("✓ Model loaded successfully!")

# Setup test data generator (same as training, but no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r'C:\Users\Administrator\Downloads\archive\test',  # Update this path to your test folder
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False  # Important: don't shuffle for evaluation
)

print(f"✓ Test data loaded: {test_data.samples} images")
print(f"✓ Classes: {list(test_data.class_indices.keys())}")

# Now evaluate
print("\n========== EVALUATING ON TEST SET ==========")
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)

print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✓ Test Loss: {test_loss:.4f}")

# Get predictions
print("\n========== GENERATING PREDICTIONS ==========")
test_data.reset()
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes
class_names = list(test_data.class_indices.keys())

print("✓ Predictions generated!")

# Generate confusion matrix
print("\n========== CONFUSION MATRIX ==========")
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix - Emotion Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# Calculate all metrics
print("\n========== CLASSIFICATION REPORT ==========")
report = classification_report(true_classes, predicted_classes, 
                              target_names=class_names, 
                              digits=4)
print(report)

# Detailed metrics
precision, recall, f1, support = precision_recall_fscore_support(
    true_classes, predicted_classes, average=None, labels=range(7)
)

# Calculate specificity
specificity = []
for i in range(len(class_names)):
    tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
    fp = np.sum(cm[:, i]) - cm[i, i]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity.append(spec)

# Create metrics table
metrics_df = pd.DataFrame({
    'Emotion': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Specificity': specificity,
    'Support': support
})

print("\n========== PER-CLASS METRICS ==========")
print(metrics_df.to_string(index=False))

# Save to CSV
metrics_df.to_csv('evaluation_metrics.csv', index=False)
print("\n✓ Metrics saved to 'evaluation_metrics.csv'")

print("\n========== EVALUATION COMPLETE ==========")