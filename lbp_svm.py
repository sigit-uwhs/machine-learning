import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# Parameter LBP
RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = "uniform"

# ===============================
# FUNGSI EKSTRAKSI LBP
# ===============================
def extract_lbp(image):
    lbp = local_binary_pattern(image, N_POINTS, RADIUS, METHOD)
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

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

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)

            features = extract_lbp(img)

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
# PIPELINE: SCALER + SVM
# ===============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])

# ===============================
# TRAINING
# ===============================
print("Training LBP + SVM...")
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
