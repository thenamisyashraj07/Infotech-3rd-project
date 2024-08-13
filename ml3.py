import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns

# Paths to your data
train_dir = r"C:\Users\Yashraj\Downloads\archive (2)\train"
test_dir = r"C:\Users\Yashraj\Downloads\archive (2)\test"

# Function to load images
def load_images_from_folder(folder):
    images = []
    labels = []
    print(f"Loading images from {folder}")
    for subdir, _, files in os.walk(folder):
        for filename in files:
            print(f"Processing file: {filename}")
            label = 1 if 'dog' in filename else 0  # 1 for dog, 0 for cat
            img_path = os.path.join(subdir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize image to 64x64
                img = img.flatten()  # Flatten the image
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Could not read image {img_path}")
    return np.array(images), np.array(labels)

# Load training data
X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)

# Check if any images were loaded
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("No images were loaded. Please check the file paths and directory structure.")

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_val_pred = svm.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

y_test_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve
y_test_prob = svm.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
