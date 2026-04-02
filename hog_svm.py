import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ===============================
# KONFIGURASI
# ===============================
IMG_SIZE = (128, 128)
DATASET_DIR = "/content/drive/MyDrive/dataset/dataset_non_augmentasi"

CLASSES = {
    "diabetes": 1,
    "nondiabetes": 0
}

# ===============================
# FUNGSI LOAD DATA
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

            # Ekstraksi HOG
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

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

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ===============================
# PIPELINE: SCALER + SVM
# ===============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

# ===============================
# OPTIONAL: TUNING PARAMETER
# ===============================
param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01, 0.001]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

print("Training model...")
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

model = grid.best_estimator_

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

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ===============================
# ROC CURVE & AUC
# ===============================

# Probabilitas kelas positif (diabetes = 1)
y_test_proba = model.predict_proba(X_test)[:, 1]

# Hitung ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # garis random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - HOG + SVM")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
