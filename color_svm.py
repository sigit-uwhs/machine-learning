import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# KONFIGURASI
# ===============================
IMG_SIZE = (128, 128)
DATASET_DIR = "/content/drive/MyDrive/dataset/dataset_non_augmentasi"

CLASSES = {
    "diabetes": 1,
    "nondiabetes": 0
}

BINS = 32  # jumlah bin histogram per channel

# ===============================
# EKSTRAKSI COLOR HISTOGRAM
# ===============================
def extract_color_histogram(image):
    image = cv2.resize(image, IMG_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=[BINS, BINS, BINS],
        ranges=[0, 180, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist.flatten()

# ===============================
# LOAD DATA
# ===============================
def load_data(split):
    X, y = [], []
    split_path = os.path.join(DATASET_DIR, split)

    for label_name, label_value in CLASSES.items():
        folder = os.path.join(split_path, label_name)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            features = extract_color_histogram(img)

            X.append(features)
            y.append(label_value)

    return np.array(X), np.array(y)

# ===============================
# LOAD DATASET
# ===============================
print("Loading training data...")
X_train, y_train = load_data("train")

print("Loading validation data...")
X_valid, y_valid = load_data("valid")

print("Loading test data...")
X_test, y_test = load_data("test")

print("Feature shape:", X_train.shape)

# ===============================
# MODEL: SCALER + SVM
# ===============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])

# ===============================
# TRAINING
# ===============================
print("Training Color Histogram + SVM...")
model.fit(X_train, y_train)

# ===============================
# VALIDATION
# ===============================
y_valid_pred = model.predict(X_valid)
print("\nValidation Accuracy:", accuracy_score(y_valid, y_valid_pred))

# ===============================
# TESTING
# ===============================
y_test_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=CLASSES.keys()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
