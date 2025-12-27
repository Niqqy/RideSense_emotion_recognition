# evaluate.py - Complete Model Evaluation Script
# This file contains ALL evaluation steps (8.1 through 8.8)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import random

print("="*60)
print("STARTING MODEL EVALUATION")
print("="*60)

# ========== LOAD TRAINED MODEL (Only once!) ==========
print("\n[1/8] Loading trained model...")
model = load_model('models/best_model.keras')
print("✓ Model loaded successfully!")

# ========== LOAD TEST DATA ==========
print("\n[2/8] Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r'C:\Users\Administrator\Downloads\archive\test',  # UPDATE THIS PATH TO YOUR TEST FOLDER
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Test data loaded: {test_data.samples} images")
class_names = list(test_data.class_indices.keys())
print(f"✓ Classes: {class_names}")

# ========== STEP 8.2: EVALUATE ON TEST SET ==========
print("\n[3/8] Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)

print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✓ Test Loss: {test_loss:.4f}")

# ========== STEP 8.3: GENERATE PREDICTIONS ==========
print("\n[4/8] Generating predictions...")
test_data.reset()
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

print("✓ Predictions generated!")

# ========== STEP 8.4: CONFUSION MATRIX ==========
print("\n[5/8] Creating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)

# Plot raw counts
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix - Emotion Recognition (Counts)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_counts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_counts.png")
plt.show()

# Plot percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix - Percentage (%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_percentage.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_percentage.png")
plt.show()

# ========== STEP 8.5: CALCULATE ALL METRICS ==========
print("\n[6/8] Calculating evaluation metrics...")

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(true_classes, predicted_classes, 
                              target_names=class_names, 
                              digits=4)
print(report)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    true_classes, predicted_classes, average=None, labels=range(7)
)

# Calculate specificity for each class
specificity = []
for i in range(len(class_names)):
    tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
    fp = np.sum(cm[:, i]) - cm[i, i]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity.append(spec)

# Create detailed metrics table
metrics_df = pd.DataFrame({
    'Emotion': class_names,
    'Precision': precision,
    'Recall (Sensitivity)': recall,
    'F1-Score': f1,
    'Specificity': specificity,
    'Support': support
})

print("\n" + "="*60)
print("PER-CLASS DETAILED METRICS")
print("="*60)
print(metrics_df.to_string(index=False))

# Save to CSV
metrics_df.to_csv('per_class_metrics.csv', index=False)
print("\n✓ Saved: per_class_metrics.csv")

# ========== STEP 8.6: MACRO AND WEIGHTED AVERAGES ==========
print("\n[7/8] Calculating summary statistics...")

# Macro averages (unweighted mean)
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)
macro_specificity = np.mean(specificity)

# Weighted averages (weighted by support)
total_support = np.sum(support)
weighted_precision = np.sum(precision * support) / total_support
weighted_recall = np.sum(recall * support) / total_support
weighted_f1 = np.sum(f1 * support) / total_support
weighted_specificity = np.sum(np.array(specificity) * support) / total_support

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nOverall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\nMacro Average (unweighted):")
print(f"  Precision:    {macro_precision:.4f}")
print(f"  Recall:       {macro_recall:.4f}")
print(f"  F1-Score:     {macro_f1:.4f}")
print(f"  Specificity:  {macro_specificity:.4f}")

print(f"\nWeighted Average (by class support):")
print(f"  Precision:    {weighted_precision:.4f}")
print(f"  Recall:       {weighted_recall:.4f}")
print(f"  F1-Score:     {weighted_f1:.4f}")
print(f"  Specificity:  {weighted_specificity:.4f}")

# Save summary statistics
summary_df = pd.DataFrame({
    'Metric': ['Overall Accuracy', 
               'Macro Precision', 'Macro Recall', 'Macro F1-Score', 'Macro Specificity',
               'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score', 'Weighted Specificity'],
    'Value': [test_accuracy,
              macro_precision, macro_recall, macro_f1, macro_specificity,
              weighted_precision, weighted_recall, weighted_f1, weighted_specificity]
})
summary_df.to_csv('summary_statistics.csv', index=False)
print("\n✓ Saved: summary_statistics.csv")

# ========== STEP 8.7: ANALYZE MISCLASSIFICATIONS ==========
print("\n[8/8] Analyzing misclassifications...")

confusion_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((class_names[i], class_names[j], cm[i, j]))

# Sort by frequency
confusion_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n" + "="*60)
print("TOP 10 MOST COMMON MISCLASSIFICATIONS")
print("="*60)
print(f"{'True Label':<12} -> {'Predicted As':<12} | Count")
print("-" * 45)
for true_label, pred_label, count in confusion_pairs[:10]:
    print(f"{true_label:<12} -> {pred_label:<12} | {count}")

# Save misclassifications
misclass_df = pd.DataFrame(confusion_pairs, 
                           columns=['True_Label', 'Predicted_As', 'Count'])
misclass_df.to_csv('misclassifications.csv', index=False)
print("\n✓ Saved: misclassifications.csv")

# ========== STEP 8.8: VISUALIZE SAMPLE PREDICTIONS ==========
print("\nVisualizing sample predictions...")

def plot_sample_predictions(num_samples=16):
    plt.figure(figsize=(16, 16))
    
    # Get random indices
    indices = random.sample(range(len(true_classes)), min(num_samples, len(true_classes)))
    
    for idx, i in enumerate(indices):
        # Get image path
        image_path = test_data.filepaths[i]
        img = plt.imread(image_path)
        
        # Get prediction details
        true_label = class_names[true_classes[i]]
        pred_label = class_names[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]] * 100
        
        # Check if correct
        is_correct = true_classes[i] == predicted_classes[i]
        color = 'green' if is_correct else 'red'
        symbol = '✓' if is_correct else '✗'
        
        # Plot
        plt.subplot(4, 4, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{symbol} True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%", 
                 color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sample_predictions.png")
    plt.show()

plot_sample_predictions(16)

# ========== SUMMARY ==========
print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print("\nGenerated Files:")
print("  1. confusion_matrix_counts.png")
print("  2. confusion_matrix_percentage.png")
print("  3. per_class_metrics.csv")
print("  4. summary_statistics.csv")
print("  5. misclassifications.csv")
print("  6. sample_predictions.png")
print("\n" + "="*60)